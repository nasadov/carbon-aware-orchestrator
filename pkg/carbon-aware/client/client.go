/*
 * Derivative work:
 * Copyright 2023 Fondazione Bruno Kessler
 *
 * Original/Previous work:
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"
	"gitlab.fbk.eu/fogatlas-k8s/algorithms/pkg/generated-go/idl"

	"k8s.io/apimachinery/pkg/api/resource"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

var (
	addr     = flag.String("addr", "localhost:50051", "the address to connect to")
	logLevel = flag.String("loglevel", "trace", "The log level")
)

type Microservice struct {
	Name         string
	Status       idl.MicroserviceStatus
	NodeSelected []string
}

func main() {
	flag.Parse()
	configureLog(*logLevel)
	// Set up a connection to the server.
	log.Info("Starting connection to server")
	conn, err := grpc.Dial(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := idl.NewPlacementAlgorithmClient(conn)

	// Contact the server and print out its response.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	infra := loadInfra()
	workload := getWorkload()
	data := idl.Data{
		Workload:       workload,
		Infrastructure: infra,
	}

	str, err := json.Marshal(&data)
	if err != nil {
		log.Errorf("error marshalling data: (%s)", err.Error())
	} else {
		log.Infof("Sending data: (%s)", str)
	}

	log.Info("Calling Init method")
	_, err = c.Init(ctx, &idl.AlgorithmName{Name: "Silly"})
	if err != nil {
		st, ok := status.FromError(err)
		if ok { // grpc error
			log.Fatalf("could not initialize algo: (%v) (%v)", st.Code(), st.Message())
		} else { // non grpc error
			log.Fatalf("could not initialize algo: %v", err)
		}
	} else {
		log.Info("Calling CalculatePlacement method")
		r, err := c.CalculatePlacement(ctx, &data)
		if err != nil {
			st, ok := status.FromError(err)
			if ok { // grpc error
				log.Fatalf("could not get placement: code=%v, desc=%v", st.Code(), st.Message())
			} else { // non grpc error
				log.Fatalf("could not get placement: %v", err)
			}
		} else {
			str, err := json.Marshal(r)
			if err != nil {
				log.Errorf("error marshalling data: (%s)", err.Error())
			} else {
				log.Infof("Received data: (%s)", str)
			}
		}
	}
}

func loadInfra() *idl.Infrastructure {
	// 4 Nodes
	var nodes []*idl.Node
	for i := 0; i < 4; i++ {
		cpu := resource.MustParse("100m")
		mem := resource.MustParse("100M")
		node := idl.Node{
			Name: fmt.Sprintf("N_%d", i),
			CpuUsed: &idl.ResourceQuantity{
				Value:  cpu.String(),
				Format: string(cpu.Format),
			},
			MemUsed: &idl.ResourceQuantity{
				Value:  mem.String(),
				Format: string(mem.Format),
			},
			CpuCap: &idl.ResourceQuantity{
				Value:  cpu.String(),
				Format: string(cpu.Format),
			},
			MemCap: &idl.ResourceQuantity{
				Value:  mem.String(),
				Format: string(mem.Format),
			},
			Region:      fmt.Sprintf("R_%d", i),
			Subcategory: fmt.Sprintf("SC_%d", i),
		}
		nodes = append(nodes, &node)
	}
	var infra idl.Infrastructure
	infra.Nodes = nodes
	return &infra
}

func getWorkload() *idl.Workload {
	var MICROSERVICES = []Microservice{
		{Name: "one", Status: idl.MicroserviceStatus_RUNNING, NodeSelected: []string{"myNode"}},
		{Name: "two", Status: idl.MicroserviceStatus_PENDING},
		{Name: "three", Status: idl.MicroserviceStatus_TO_SCHEDULE},
		{Name: "four", Status: idl.MicroserviceStatus_TO_DEPLOY},
	}
	var workload idl.Workload
	var msList []*idl.Microservice
	// 4 microservices: one running, one pending, one to be scheduled and one to be deployed
	for i := 0; i < 4; i++ {
		cpu := resource.MustParse("100m")
		mem := resource.MustParse("100M")
		microservice := idl.Microservice{
			Name:       MICROSERVICES[i].Name,
			Status:     MICROSERVICES[i].Status,
			DeployedOn: MICROSERVICES[i].NodeSelected,
			Replicas:   int32(1),
			CpuRequired: &idl.ResourceQuantity{
				Value:  cpu.String(),
				Format: string(cpu.Format),
			},
			MemRequired: &idl.ResourceQuantity{
				Value:  mem.String(),
				Format: string(mem.Format),
			},
		}
		msList = append(msList, &microservice)
	}

	workload.Microservices = msList
	return &workload
}

// configureLog function
func configureLog(logLevel string) {
	var loggerLevel log.Level
	switch logLevel {
	case "trace":
		loggerLevel = log.TraceLevel
	case "debug":
		loggerLevel = log.DebugLevel
	case "info":
		loggerLevel = log.InfoLevel
	case "warn":
		loggerLevel = log.WarnLevel
	case "error":
		loggerLevel = log.ErrorLevel
	case "fatal":
		loggerLevel = log.FatalLevel
	case "panic":
		loggerLevel = log.PanicLevel
	default:
		loggerLevel = log.InfoLevel
	}

	log.SetLevel(loggerLevel)
	log.SetFormatter(&log.TextFormatter{
		DisableColors: false,
		FullTimestamp: true,
	})
}
