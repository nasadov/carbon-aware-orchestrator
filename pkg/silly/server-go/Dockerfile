#
# Copyright 2023 Fondazione Bruno Kessler.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Setup GO builder container
FROM golang as builder

WORKDIR /go/src/gitlab.fbk.eu/fogatlas/algorithms

COPY ./ .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o silly pkg/silly/server-go/silly.go

# Build application container from GO artifact
FROM scratch
WORKDIR /

COPY --from=builder /go/src/gitlab.fbk.eu/fogatlas/algorithms/silly /bin/silly

ENTRYPOINT ["silly"]
