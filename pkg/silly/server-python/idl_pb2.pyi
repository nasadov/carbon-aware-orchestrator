from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MicroserviceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MICROSERVICESTATUS_UNSPECIFIED: _ClassVar[MicroserviceStatus]
    RUNNING: _ClassVar[MicroserviceStatus]
    PENDING: _ClassVar[MicroserviceStatus]
    TO_SCHEDULE: _ClassVar[MicroserviceStatus]
    TO_DEPLOY: _ClassVar[MicroserviceStatus]
MICROSERVICESTATUS_UNSPECIFIED: MicroserviceStatus
RUNNING: MicroserviceStatus
PENDING: MicroserviceStatus
TO_SCHEDULE: MicroserviceStatus
TO_DEPLOY: MicroserviceStatus

class AlgorithmName(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ResourceQuantity(_message.Message):
    __slots__ = ("value", "format")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    value: str
    format: str
    def __init__(self, value: _Optional[str] = ..., format: _Optional[str] = ...) -> None: ...

class Data(_message.Message):
    __slots__ = ("workload", "infrastructure")
    WORKLOAD_FIELD_NUMBER: _ClassVar[int]
    INFRASTRUCTURE_FIELD_NUMBER: _ClassVar[int]
    workload: Workload
    infrastructure: Infrastructure
    def __init__(self, workload: _Optional[_Union[Workload, _Mapping]] = ..., infrastructure: _Optional[_Union[Infrastructure, _Mapping]] = ...) -> None: ...

class Infrastructure(_message.Message):
    __slots__ = ("nodes",)
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ("name", "cpu_used", "mem_used", "cpu_cap", "mem_cap", "region", "subcategory")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CPU_USED_FIELD_NUMBER: _ClassVar[int]
    MEM_USED_FIELD_NUMBER: _ClassVar[int]
    CPU_CAP_FIELD_NUMBER: _ClassVar[int]
    MEM_CAP_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    SUBCATEGORY_FIELD_NUMBER: _ClassVar[int]
    name: str
    cpu_used: ResourceQuantity
    mem_used: ResourceQuantity
    cpu_cap: ResourceQuantity
    mem_cap: ResourceQuantity
    region: str
    subcategory: str
    def __init__(self, name: _Optional[str] = ..., cpu_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_used: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., cpu_cap: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_cap: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., region: _Optional[str] = ..., subcategory: _Optional[str] = ...) -> None: ...

class Workload(_message.Message):
    __slots__ = ("microservices",)
    MICROSERVICES_FIELD_NUMBER: _ClassVar[int]
    microservices: _containers.RepeatedCompositeFieldContainer[Microservice]
    def __init__(self, microservices: _Optional[_Iterable[_Union[Microservice, _Mapping]]] = ...) -> None: ...

class Microservice(_message.Message):
    __slots__ = ("name", "status", "deployed_on", "replicas", "cpu_required", "mem_required", "duration_hours", "deadline_hours", "annotations")
    NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    DEPLOYED_ON_FIELD_NUMBER: _ClassVar[int]
    REPLICAS_FIELD_NUMBER: _ClassVar[int]
    CPU_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    MEM_REQUIRED_FIELD_NUMBER: _ClassVar[int]
    DURATION_HOURS_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_HOURS_FIELD_NUMBER: _ClassVar[int]
    ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    name: str
    status: MicroserviceStatus
    deployed_on: _containers.RepeatedScalarFieldContainer[str]
    replicas: int
    cpu_required: ResourceQuantity
    mem_required: ResourceQuantity
    duration_hours: float
    deadline_hours: float
    annotations: _containers.RepeatedCompositeFieldContainer[KeyValue]
    def __init__(self, name: _Optional[str] = ..., status: _Optional[_Union[MicroserviceStatus, str]] = ..., deployed_on: _Optional[_Iterable[str]] = ..., replicas: _Optional[int] = ..., cpu_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., mem_required: _Optional[_Union[ResourceQuantity, _Mapping]] = ..., duration_hours: _Optional[float] = ..., deadline_hours: _Optional[float] = ..., annotations: _Optional[_Iterable[_Union[KeyValue, _Mapping]]] = ...) -> None: ...

class KeyValue(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class Placements(_message.Message):
    __slots__ = ("placements",)
    PLACEMENTS_FIELD_NUMBER: _ClassVar[int]
    placements: _containers.RepeatedCompositeFieldContainer[Placement]
    def __init__(self, placements: _Optional[_Iterable[_Union[Placement, _Mapping]]] = ...) -> None: ...

class Placement(_message.Message):
    __slots__ = ("microservice_name", "replica_scores", "time_to_schedule")
    MICROSERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    REPLICA_SCORES_FIELD_NUMBER: _ClassVar[int]
    TIME_TO_SCHEDULE_FIELD_NUMBER: _ClassVar[int]
    microservice_name: str
    replica_scores: _containers.RepeatedCompositeFieldContainer[ReplicaScores]
    time_to_schedule: int
    def __init__(self, microservice_name: _Optional[str] = ..., replica_scores: _Optional[_Iterable[_Union[ReplicaScores, _Mapping]]] = ..., time_to_schedule: _Optional[int] = ...) -> None: ...

class ReplicaScores(_message.Message):
    __slots__ = ("scores",)
    SCORES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedCompositeFieldContainer[Score]
    def __init__(self, scores: _Optional[_Iterable[_Union[Score, _Mapping]]] = ...) -> None: ...

class Score(_message.Message):
    __slots__ = ("node", "score")
    NODE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    node: str
    score: int
    def __init__(self, node: _Optional[str] = ..., score: _Optional[int] = ...) -> None: ...
