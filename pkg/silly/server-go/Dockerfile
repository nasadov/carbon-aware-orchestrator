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
