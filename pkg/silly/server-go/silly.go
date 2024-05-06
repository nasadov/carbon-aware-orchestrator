/*
 *Copyright 2023 Fondazione Bruno Kessler.
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *
 */

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net"

	log "github.com/sirupsen/logrus"
	"gitlab.fbk.eu/fogatlas-k8s/algorithms/pkg/generated-go/idl"
	"k8s.io/apimachinery/pkg/api/resource"

	empty "github.com/golang/protobuf/ptypes/empty"

	"math"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var (
	port     = flag.Int("port", 50051, "The server port")
	logLevel = flag.String("loglevel", "trace", "The log level")
)

type server struct {
	idl.UnimplementedPlacementAlgorithmServer
}

// Algorithm contains algorithm's name and flags
type Algorithm struct {
	name        string
	initialized bool
}

var algo Algorithm

// Init method
func (s *server) Init(ctx context.Context, in *idl.AlgorithmName) (*empty.Empty, error) {
	log.Tracef("Init: received: %v", in.GetName())
	algo.name = in.GetName()
	algo.initialized = true
	return &empty.Empty{}, nil
}

// CalculatePlacement method
//
// It just return a score per node per microservice according to the order of the nodes:
// score = (indexNode + 1)
func (s *server) CalculatePlacement(ctx context.Context, in *idl.Data) (*idl.Workload, error) {
	str, err := json.Marshal(&in)
	if err != nil {
		log.Errorf("error marshalling data: (%s)", err.Error())
	} else {
		log.Tracef("CalculatePlacement - Received data: (%s)", str)
	}

	log.Infof("Going to calculate the placement of with algorithm %s", algo.name)

	if !algo.initialized {
		log.Errorf("SillyAlgorithm not intitialized.")
		return nil, status.Errorf(codes.FailedPrecondition, "silly algorithm not intitialized.")
	}

	inInfra := in.GetInfrastructure()
	inWorkload := in.GetWorkload()
	outWorkload := new(idl.Workload)

	// Get node list (do not consider regions)
	var nodeList []*idl.Node

	for _, reg := range inInfra.GetRegions() {
		nodeList = append(nodeList, reg.GetNodes()...)
	}

	// An example on how to convert a pd.ResourceQuantity to a resource.Quantity
	node := nodeList[0]
	q, _ := resource.ParseQuantity(node.CpuUsed.Value)
	log.Tracef("Node cpu used is: (%s)", q.String())

	for j, ms := range inWorkload.GetMicroservices() {
		placement := new(idl.Placement)
		placement.MicroserviceName = ms.GetName()
		placement.Order = int32(len(inWorkload.GetMicroservices()) - j)
		// Assume just one replica
		replicaScore := new(idl.ReplicaScores)
		for k, node := range nodeList {
			score := new(idl.Score)
			score.Node = node.Name
			score.Score = int32(math.Pow(float64(k+1), 3))
			replicaScore.Scores = append(replicaScore.Scores, score)
		}
		placement.ReplicaScores = append(placement.ReplicaScores, replicaScore)
		outWorkload.Placements = append(outWorkload.Placements, placement)
	}

	strout, err := json.Marshal(&outWorkload)
	if err != nil {
		log.Errorf("error marshalling data: (%s)", err.Error())
	} else {
		log.Tracef("CalculatePlacement - Returning data: (%s)", strout)
	}
	return outWorkload, nil
}

func main() {
	flag.Parse()
	configureLog(*logLevel)
	log.Trace("This is a test for log level")
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	s := grpc.NewServer()
	idl.RegisterPlacementAlgorithmServer(s, &server{})
	log.Infof("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
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
