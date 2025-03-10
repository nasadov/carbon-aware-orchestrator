/*
 * Derivative work:
 * Copyright 2025 Technical University of Berlin
 *
 * Previous derivative work:
 * Copyright 2023 Fondazione Bruno Kessler
 *
 * Original work:
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
	"strconv"
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
	Name          string
	Status        idl.MicroserviceStatus
	NodeSelected  []string
	Duration      string // Duration in hours
	Deadline      string // Deadline in hours
	CpuRequest    string // CPU request in cores or millicores
	MemoryRequest string // Memory request in bytes (Ki, Mi, Gi)
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
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30) // Longer timeout
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
	// Define proper regions and hardware subcategories
	regions := []string{"DE", "FR", "ES", "IT-NO"}
	subcategories := []string{"IoT", "Server", "Laptop", "Smartphone"}

	var nodes []*idl.Node
	for i := 0; i < 8; i++ { // Create 8 nodes like infra_workload_gen.py
		// Set CPU/memory capacity based on hardware type
		var cpuCap, memCap resource.Quantity
		subcategory := subcategories[i%len(subcategories)]

		switch subcategory {
		case "IoT":
			cpuCap = resource.MustParse("2")
			memCap = resource.MustParse("2Gi")
		case "Smartphone":
			cpuCap = resource.MustParse("4")
			memCap = resource.MustParse("4Gi")
		case "Laptop":
			cpuCap = resource.MustParse("8")
			memCap = resource.MustParse("16Gi")
		case "Server":
			cpuCap = resource.MustParse("32")
			memCap = resource.MustParse("64Gi")
		}

		// All nodes start with zero usage
		cpuUsedVal := float64(0) // Zero CPU usage
		memUsedVal := float64(0) // Zero memory usage

		cpuUsed := resource.NewQuantity(int64(cpuUsedVal), cpuCap.Format)
		memUsed := resource.NewQuantity(int64(memUsedVal), memCap.Format)

		region := regions[i%len(regions)]
		nodeName := fmt.Sprintf("node-%d-%s-%s", i, region, subcategory)

		node := idl.Node{
			Name: nodeName,
			CpuUsed: &idl.ResourceQuantity{
				Value:  cpuUsed.String(),
				Format: string(cpuUsed.Format),
			},
			MemUsed: &idl.ResourceQuantity{
				Value:  memUsed.String(),
				Format: string(memUsed.Format),
			},
			CpuCap: &idl.ResourceQuantity{
				Value:  cpuCap.String(),
				Format: string(cpuCap.Format),
			},
			MemCap: &idl.ResourceQuantity{
				Value:  memCap.String(),
				Format: string(memCap.Format),
			},
			Region:      region,
			Subcategory: subcategory,
		}
		nodes = append(nodes, &node)
	}
	var infra idl.Infrastructure
	infra.Nodes = nodes
	return &infra
}

func getWorkload() *idl.Workload {
	var MICROSERVICES = []Microservice{
		{
			Name:          "one",
			Status:        idl.MicroserviceStatus_TO_DEPLOY,
			Duration:      "3",
			Deadline:      "24", // 24h deadline - much more than duration
			CpuRequest:    "500m",
			MemoryRequest: "1Gi",
		},
		{
			Name:          "two",
			Status:        idl.MicroserviceStatus_TO_DEPLOY,
			Duration:      "6",
			Deadline:      "12", // 12h deadline - some flexibility
			CpuRequest:    "1000m",
			MemoryRequest: "2Gi",
		},
		{
			Name:          "three",
			Status:        idl.MicroserviceStatus_TO_DEPLOY,
			Duration:      "1",
			Deadline:      "3", // 3h deadline - limited flexibility
			CpuRequest:    "250m",
			MemoryRequest: "512Mi",
		},
		{
			Name:          "four",
			Status:        idl.MicroserviceStatus_RUNNING,
			NodeSelected:  []string{"node-0-DE-IoT"}, // Updated node name format
			Duration:      "12",
			Deadline:      "12", // 12h deadline - no flexibility
			CpuRequest:    "100m",
			MemoryRequest: "256Mi",
		},
	}

	var workload idl.Workload
	var msList []*idl.Microservice

	for _, ms := range MICROSERVICES {
		cpu := resource.MustParse(ms.CpuRequest)
		mem := resource.MustParse(ms.MemoryRequest)

		// Parse duration and deadline
		durFloat, _ := strconv.ParseFloat(ms.Duration, 64)
		deadlineFloat, _ := strconv.ParseFloat(ms.Deadline, 64)

		// Encode both duration and deadline in the name
		name := fmt.Sprintf("%s-duration-%sh-deadline-%sh",
			ms.Name, ms.Duration, ms.Deadline)

		// Build the microservice with all scheduling information
		// using direct fields instead of annotations
		microservice := idl.Microservice{
			Name:       name,
			Status:     ms.Status,
			DeployedOn: ms.NodeSelected,
			Replicas:   int32(1),
			CpuRequired: &idl.ResourceQuantity{
				Value:  cpu.String(),
				Format: string(cpu.Format),
			},
			MemRequired: &idl.ResourceQuantity{
				Value:  mem.String(),
				Format: string(mem.Format),
			},
			// Direct fields for scheduling parameters
			DurationHours: durFloat,
			DeadlineHours: deadlineFloat,
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
