from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AlgorithmName(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class Application(_message.Message):
    __slots__ = ["data_flows", "external_endpoints", "microservices", "name", "order", "placements"]
    DATA_FLOWS_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_ENDPOINTS_FIELD_NUMBER: _ClassVar[int]
    MICROSERVICES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    data_flows: _containers.RepeatedCompositeFieldContainer[DataFlow]
    external_endpoints: _containers.RepeatedScalarFieldContainer[str]
    microservices: _containers.RepeatedCompositeFieldContainer[Microservice]
    name: str
    order: int
    placements: _containers.RepeatedCompositeFieldContainer[Placement]
    def __init__(self, name: _Optional[str] = ..., order: _Optional[int] = ..., external_endpoints: _Optional[_Iterable[str]] = ..., microservices: _Optional[_Iterable[_Union[Microservice, _Mapping]]] = ..., data_flows: _Optional[_Iterable[_Union[DataFlow, _Mapping]]] = ..., placements: _Optional[_Iterable[_Union[Placement, _Mapping]]] = ...) -> None: ...

class Data(_message.Message):
    __slots__ = ["infrastructure", "workload"]
    INFRASTRUCTURE_FIELD_NUMBER: _ClassVar[int]
    WORKLOAD_FIELD_NUMBER: _ClassVar[int]
    infrastructure: Infrastructure
    workload: Workload
    def __init__(self, workload: _Optional[_Union[Workload, _Mapping]] = ..., infrastructure: _Optional[_Union[Infrastructure, _Mapping]] = ...) -> None: ...

class DataFlow(_message.Message):
    __slots__ = ["bandwidth_required", "latency_required", "name", "prev_bandwidth_required", "vertices"]
    BANDWIDTH_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    LATENCY_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PREV_BANDWIDTH_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    VERTICES_FIELD_NUMBER: _ClassVar[int]
    bandwidth_required: ResourceQuantity
    latency_required: ResourceQuantity
    name: str
    prev_bandwidth_required: ResourceQuantity
    vertices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., prev_bandwidth_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., bandwidth_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., latency_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., vertices: _Optional[_Iterable[str]] = ...) -> None: ...

class Infrastructure(_message.Message):
    __slots__ = ["links", "regions"]
    LINKS_FIELD_NUMBER: _ClassVar[int]
    REGIONS_FIELD_NUMBER: _ClassVar[int]
    links: _containers.RepeatedCompositeFieldContainer[Link]
    regions: _containers.RepeatedCompositeFieldContainer[Region]
    def __init__(self, regions: _Optional[_Iterable[_Union[Region, _Mapping]]] = ..., links: _Optional[_Iterable[_Union[Link, _Mapping]]] = ...) -> None: ...

class Link(_message.Message):
    __slots__ = ["bandwidth", "bandwidth_used", "endpoint_a", "endpoint_b", "id", "latency"]
    BANDWIDTH_FIELD_NUMBER: _ClassVar[int]
    BANDWIDTH_USED_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_A_FIELD_NUMBER: _ClassVar[int]
    ENDPOINT_B_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LATENCY_FIELD_NUMBER: _ClassVar[int]
    bandwidth: ResourceQuantity
    bandwidth_used: ResourceQuantity
    endpoint_a: str
    endpoint_b: str
    id: str
    latency: ResourceQuantity
    def __init__(self, id: _Optional[str] = ..., endpoint_a: _Optional[str] = ..., endpoint_b: _Optional[str] = ..., bandwidth: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., latency: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., bandwidth_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ...) -> None: ...

class Microservice(_message.Message):
    __slots__ = ["budget_4_1_reschedule", "cpu_required", "disruption_budget", "mem_required", "name", "no_reschedule", "prev_cpu_required", "prev_mem_required", "region_recommended", "replicas"]
    BUDGET_4_1_RESCHEDULE_FIELD_NUMBER: _ClassVar[int]
    CPU_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DISRUPTION_BUDGET_FIELD_NUMBER: _ClassVar[int]
    MEM_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NO_RESCHEDULE_FIELD_NUMBER: _ClassVar[int]
    PREV_CPU_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    PREV_MEM_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    REGION_RECOMMENDED_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    budget_4_1_reschedule: int
    cpu_required: ResourceQuantity
    disruption_budget: int
    mem_required: ResourceQuantity
    name: str
    no_reschedule: bool
    prev_cpu_required: ResourceQuantity
    prev_mem_required: ResourceQuantity
    region_recommended: str
    replicas: int
    def __init__(self, name: _Optional[str] = ..., region_recommended: _Optional[str] = ..., no_reschedule: bool = ..., disruption_budget: _Optional[int] = ..., budget_4_1_reschedule: _Optional[int] = ..., replicas: _Optional[int] = ..., prev_cpu_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., prev_mem_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., cpu_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ["cpu_cap", "cpu_used", "mem_cap", "mem_used", "name"]
    CPU_CAP_FIELD_NUMBER: _ClassVar[int]
    CPU_USED_FIELD_NUMBER: _ClassVar[int]
    MEM_CAP_FIELD_NUMBER: _ClassVar[int]
    MEM_USED_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    cpu_cap: ResourceQuantity
    cpu_used: ResourceQuantity
    mem_cap: ResourceQuantity
    mem_used: ResourceQuantity
    name: str
    def __init__(self, name: _Optional[str] = ..., cpu_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., cpu_cap: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_cap: _Optional[_Union[ResourceQuantity, _Mapping]] = ...) -> None: ...

class Placement(_message.Message):
    __slots__ = ["microservice_name", "must_reschedule", "node_selected", "order", "replica_scores"]
    MICROSERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    MUST_RESCHEDULE_FIELD_NUMBER: _ClassVar[int]
    NODE_SELECTED_FIELD_NUMBER: _ClassVar[int]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    REPLICA_SCORES_FIELD_NUMBER: _ClassVar[int]
    microservice_name: str
    must_reschedule: bool
    node_selected: _containers.RepeatedScalarFieldContainer[str]
    order: int
    replica_scores: _containers.RepeatedCompositeFieldContainer[ReplicaScores]
    def __init__(self, microservice_name: _Optional[str] = ..., order: _Optional[int] = ..., must_reschedule: bool = ..., replica_scores: _Optional[_Iterable[_Union[ReplicaScores, _Mapping]]] = ..., node_selected: _Optional[_Iterable[str]] = ...) -> None: ...

class Region(_message.Message):
    __slots__ = ["extendpoint_ids", "id", "location", "nodes", "tier"]
    EXTENDPOINT_IDS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    extendpoint_ids: _containers.RepeatedScalarFieldContainer[str]
    id: str
    location: str
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    tier: int
    def __init__(self, id: _Optional[str] = ..., location: _Optional[str] = ..., tier: _Optional[int] = ..., extendpoint_ids: _Optional[_Iterable[str]] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class ReplicaScores(_message.Message):
    __slots__ = ["scores"]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedCompositeFieldContainer[Score]
    def __init__(self, scores: _Optional[_Iterable[_Union[Score, _Mapping]]] = ...) -> None: ...

class ResourceQuantity(_message.Message):
    __slots__ = ["format", "value"]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    format: str
    value: str
    def __init__(self, value: _Optional[str] = ..., format: _Optional[str] = ...) -> None: ...

class Score(_message.Message):
    __slots__ = ["node", "score"]
    NODE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    node: str
    score: int
    def __init__(self, node: _Optional[str] = ..., score: _Optional[int] = ...) -> None: ...

class Workload(_message.Message):
    __slots__ = ["applications"]
    APPLICATIONS_FIELD_NUMBER: _ClassVar[int]
    applications: _containers.RepeatedCompositeFieldContainer[Application]
    def __init__(self, applications: _Optional[_Iterable[_Union[Application, _Mapping]]] = ...) -> None: ...
