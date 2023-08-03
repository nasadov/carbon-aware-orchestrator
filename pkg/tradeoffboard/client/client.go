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
	"time"

	log "github.com/sirupsen/logrus"
	"k8s.io/apimachinery/pkg/api/resource"

	idl "gitlab.fbk.eu/fogatlas-k8s/algorithms/pkg/generated-go/idl"

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

	_, err = c.Init(ctx, &idl.AlgorithmName{Name: "Tradeoffboard"})
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

	linkonpremiseedge := new(idl.Link)
	linkonpremiseedge.EndpointA = "onpremiseregion"
	linkonpremiseedge.EndpointB = "edgeregion"
	lat := resource.MustParse("10m")
	bw := resource.MustParse("100M")
	var usedBW resource.Quantity
	linkonpremiseedge.Latency = &idl.ResourceQuantity{
		Value:  lat.String(),
		Format: string(lat.Format),
	}
	linkonpremiseedge.Bandwidth = &idl.ResourceQuantity{
		Value:  bw.String(),
		Format: string(bw.Format),
	}
	linkonpremiseedge.BandwidthUsed = &idl.ResourceQuantity{
		Value:  usedBW.String(),
		Format: string(usedBW.Format),
	}
	linkonpremisecloud := new(idl.Link)
	linkonpremisecloud.EndpointA = "onpremiseregion"
	linkonpremisecloud.EndpointB = "cloudregion"
	lat = resource.MustParse("100m")
	bw = resource.MustParse("100M")
	linkonpremisecloud.Latency = &idl.ResourceQuantity{
		Value:  lat.String(),
		Format: string(lat.Format),
	}
	linkonpremisecloud.Bandwidth = &idl.ResourceQuantity{
		Value:  bw.String(),
		Format: string(bw.Format),
	}
	linkonpremisecloud.BandwidthUsed = &idl.ResourceQuantity{
		Value:  usedBW.String(),
		Format: string(usedBW.Format),
	}
	linkonedgecloud := new(idl.Link)
	linkonedgecloud.EndpointA = "edgeregion"
	linkonedgecloud.EndpointB = "cloudregion"
	lat = resource.MustParse("100m")
	bw = resource.MustParse("10M")
	linkonedgecloud.Latency = &idl.ResourceQuantity{
		Value:  lat.String(),
		Format: string(lat.Format),
	}
	linkonedgecloud.Bandwidth = &idl.ResourceQuantity{
		Value:  bw.String(),
		Format: string(bw.Format),
	}
	linkonedgecloud.BandwidthUsed = &idl.ResourceQuantity{
		Value:  usedBW.String(),
		Format: string(usedBW.Format),
	}
	links = append(links, linkonpremisecloud)
	links = append(links, linkonpremiseedge)
	links = append(links, linkonedgecloud)

	var nodesonpremise []*idl.Node
	var nodesedge []*idl.Node
	var nodescloud []*idl.Node
	cpuUsed := resource.MustParse("1000m")
	memUsed := resource.MustParse("1G")
	cpuCap := resource.MustParse("4")
	memCap := resource.MustParse("4G")

	nodeonpremise := idl.Node{
		Name: "onpremisenode",
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
	}
	cpuUsed = resource.MustParse("100m")
	memUsed = resource.MustParse("100M")
	cpuCap = resource.MustParse("1")
	memCap = resource.MustParse("1G")
	nodeedge := idl.Node{
		Name: "edgenode",
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
	}
	cpuUsed = resource.MustParse("2")
	memUsed = resource.MustParse("2G")
	cpuCap = resource.MustParse("8")
	memCap = resource.MustParse("8G")
	nodecloud := idl.Node{
		Name: "cloudnode",
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
	}
	nodesonpremise = append(nodesonpremise, &nodeonpremise)
	nodesedge = append(nodesedge, &nodeedge)
	nodescloud = append(nodescloud, &nodecloud)

	var regions []*idl.Region

	regiononpremise := idl.Region{
		Id:       "onpremiseregion",
		Location: "onpremiselocation",
		Tier:     int32(2),
		Nodes:    nodesonpremise,
	}
	regiononedge := idl.Region{
		Id:       "edgeregion",
		Location: "edgelocation",
		Tier:     int32(3),
		Nodes:    nodesedge,
	}
	regioncloud := idl.Region{
		Id:       "cloudregion",
		Location: "cloudlocation",
		Tier:     int32(1),
		Nodes:    nodescloud,
	}
	regions = append(regions, &regiononpremise)
	regions = append(regions, &regiononedge)
	regions = append(regions, &regioncloud)

	var infra idl.Infrastructure
	infra.Links = links
	infra.Regions = regions
	return &infra
}

func getWorkload() *idl.Workload {

	var workload idl.Workload

	var msList []*idl.Microservice
	cpu := resource.MustParse("1")
	mem := resource.MustParse("1G")
	microservicePR := idl.Microservice{
		Name:         "cryptoac-proxy",
		NoReschedule: false,
		CpuRequired: &idl.ResourceQuantity{
			Value:  cpu.String(),
			Format: string(cpu.Format),
		},
		MemRequired: &idl.ResourceQuantity{
			Value:  mem.String(),
			Format: string(mem.Format),
		},
	}
	cpu = resource.MustParse("100m")
	mem = resource.MustParse("100M")
	microserviceMM := idl.Microservice{
		Name:         "cryptoac-redis",
		NoReschedule: false,
		CpuRequired: &idl.ResourceQuantity{
			Value:  cpu.String(),
			Format: string(cpu.Format),
		},
		MemRequired: &idl.ResourceQuantity{
			Value:  mem.String(),
			Format: string(mem.Format),
		},
	}
	cpu = resource.MustParse("2")
	mem = resource.MustParse("2G")
	microserviceDM := idl.Microservice{
		Name:         "cryptoac-dm",
		NoReschedule: false,
		CpuRequired: &idl.ResourceQuantity{
			Value:  cpu.String(),
			Format: string(cpu.Format),
		},
		MemRequired: &idl.ResourceQuantity{
			Value:  mem.String(),
			Format: string(mem.Format),
		},
	}
	cpu = resource.MustParse("2")
	mem = resource.MustParse("2G")
	microserviceRM := idl.Microservice{
		Name:         "cryptoac-rm",
		NoReschedule: false,
		CpuRequired: &idl.ResourceQuantity{
			Value:  cpu.String(),
			Format: string(cpu.Format),
		},
		MemRequired: &idl.ResourceQuantity{
			Value:  mem.String(),
			Format: string(mem.Format),
		},
	}
	msList = append(msList, &microservicePR)
	msList = append(msList, &microserviceMM)
	msList = append(msList, &microserviceDM)
	msList = append(msList, &microserviceRM)

	var dfList []*idl.DataFlow
	lat := resource.MustParse("50m")
	bw := resource.MustParse("80M")
	dfPRMM := idl.DataFlow{
		Name: "dfPRMM",
		BandwidthRequired: &idl.ResourceQuantity{
			Value:  bw.String(),
			Format: string(bw.Format),
		},
		LatencyRequired: &idl.ResourceQuantity{
			Value:  lat.String(),
			Format: string(lat.Format),
		},
		Vertices: []string{"cryptoac-proxy", "cryptoac-redis"},
	}
	lat = resource.MustParse("500m")
	bw = resource.MustParse("50M")
	dfPRDM := idl.DataFlow{
		Name: "dfPRDM",
		BandwidthRequired: &idl.ResourceQuantity{
			Value:  bw.String(),
			Format: string(bw.Format),
		},
		LatencyRequired: &idl.ResourceQuantity{
			Value:  lat.String(),
			Format: string(lat.Format),
		},
		Vertices: []string{"cryptoac-proxy", "cryptoac-dm"},
	}
	lat = resource.MustParse("500m")
	bw = resource.MustParse("50M")
	dfRMDM := idl.DataFlow{
		Name: "dfRMDM",
		BandwidthRequired: &idl.ResourceQuantity{
			Value:  bw.String(),
			Format: string(bw.Format),
		},
		LatencyRequired: &idl.ResourceQuantity{
			Value:  lat.String(),
			Format: string(lat.Format),
		},
		Vertices: []string{"cryptoac-rm", "cryptoac-dm"},
	}
	lat = resource.MustParse("500m")
	bw = resource.MustParse("5M")
	dfRMMM := idl.DataFlow{
		Name: "dfRMMM",
		BandwidthRequired: &idl.ResourceQuantity{
			Value:  bw.String(),
			Format: string(bw.Format),
		},
		LatencyRequired: &idl.ResourceQuantity{
			Value:  lat.String(),
			Format: string(lat.Format),
		},
		Vertices: []string{"cryptoac-rm", "cryptoac-redis"},
	}
	dfList = append(dfList, &dfPRMM)
	dfList = append(dfList, &dfPRDM)
	dfList = append(dfList, &dfRMDM)
	dfList = append(dfList, &dfRMMM)

	application := idl.Application{
		Name:          "CryptoAC",
		Microservices: msList,
		DataFlows:     dfList,
	}
	workload.Applications = append(workload.Applications, &application)

	return &workload
}
