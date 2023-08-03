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
	addr = flag.String("addr", "localhost:50051", "the address to connect to")
)

func main() {
	flag.Parse()
	// Set up a connection to the server.
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

	_, err = c.Init(ctx, &idl.AlgorithmName{Name: "Silly"})
	if err != nil {
		st, ok := status.FromError(err)
		if ok { // grpc error
			log.Fatalf("could not initialize algo: (%v) (%v)", st.Code(), st.Message())
		} else { // non grpc error
			log.Fatalf("could not initialize algo: %v", err)
		}
	} else {
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
	var links []*idl.Link
	for i := 0; i < 3; i++ {
		link := new(idl.Link)
		link.EndpointA = fmt.Sprintf("A_%d", i)
		link.EndpointB = fmt.Sprintf("B_%d", i)
		lat := resource.MustParse("100m")
		bw := resource.MustParse("100M")
		link.Latency = &idl.ResourceQuantity{
			Value:  lat.String(),
			Format: string(lat.Format),
		}
		link.Bandwidth = &idl.ResourceQuantity{
			Value:  bw.String(),
			Format: string(bw.Format),
		}
		link.BandwidthUsed = &idl.ResourceQuantity{
			Value:  bw.String(),
			Format: string(bw.Format),
		}
		links = append(links, link)
	}

	var nodes []*idl.Node
	for i := 0; i < 3; i++ {
		cpu := resource.MustParse("100m")
		mem := resource.MustParse("100M")
		node := idl.Node{
			Name: fmt.Sprintf("A_%d", i),
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
		}
		nodes = append(nodes, &node)
	}

	var regions []*idl.Region
	for i := 0; i < 3; i++ {
		var extIds []string

		for i := 0; i < 3; i++ {
			extep := fmt.Sprintf("E_%d", i)
			extIds = append(extIds, extep)
		}
		region := idl.Region{
			Id:             fmt.Sprintf("R_%d", i),
			Location:       fmt.Sprintf("L_%d", i),
			Tier:           int32(i),
			ExtendpointIds: extIds,
			Nodes:          nodes,
		}
		regions = append(regions, &region)
	}
	var infra idl.Infrastructure
	infra.Links = links
	infra.Regions = regions
	return &infra
}

func getWorkload() *idl.Workload {
	var workload idl.Workload
	var msList []*idl.Microservice
	for i := 0; i < 3; i++ {
		cpu := resource.MustParse("100m")
		mem := resource.MustParse("100M")
		microservice := idl.Microservice{
			Name:                 fmt.Sprintf("M_%d", i),
			RegionRecommended:    fmt.Sprintf("R_%d", i),
			NoReschedule:         false,
			DisruptionBudget:     int32(i),
			Budget_4_1Reschedule: int32(i),
			Replicas:             int32(i),
			CpuRequired: &idl.ResourceQuantity{
				Value:  cpu.String(),
				Format: string(cpu.Format),
			},
			MemRequired: &idl.ResourceQuantity{
				Value:  mem.String(),
				Format: string(mem.Format),
			},
			PrevCpuRequired: &idl.ResourceQuantity{
				Value:  cpu.String(),
				Format: string(cpu.Format),
			},
			PrevMemRequired: &idl.ResourceQuantity{
				Value:  mem.String(),
				Format: string(mem.Format),
			},
		}
		msList = append(msList, &microservice)
	}

	var dfList []*idl.DataFlow
	for i := 0; i < 3; i++ {
		lat := resource.MustParse("100m")
		bw := resource.MustParse("100M")
		df := idl.DataFlow{
			Name: fmt.Sprintf("D_%d", i),
			BandwidthRequired: &idl.ResourceQuantity{
				Value:  bw.String(),
				Format: string(bw.Format),
			},
			PrevBandwidthRequired: &idl.ResourceQuantity{
				Value:  bw.String(),
				Format: string(bw.Format),
			},
			LatencyRequired: &idl.ResourceQuantity{
				Value:  lat.String(),
				Format: string(lat.Format),
			},
			Vertices: []string{fmt.Sprintf("S_%d", i), fmt.Sprintf("D_%d", i)},
		}
		dfList = append(dfList, &df)
	}

	var replicaScoreList []*idl.ReplicaScores
	var placementList []*idl.Placement
	for i := 0; i < 3; i++ {
		placement := idl.Placement{
			MicroserviceName: fmt.Sprintf("M_%d", i),
			MustReschedule:   false,
			ReplicaScores:    replicaScoreList,
			NodeSelected:     []string{"A", "B", "C"},
		}
		placementList = append(placementList, &placement)
	}

	for i := 0; i < 3; i++ {
		application := idl.Application{
			Name:              fmt.Sprintf("A_%d", i),
			ExternalEndpoints: []string{"A", "B", "C"},
			Microservices:     msList,
			DataFlows:         dfList,
			Placements:        placementList,
		}
		workload.Applications = append(workload.Applications, &application)
	}
	return &workload
}
