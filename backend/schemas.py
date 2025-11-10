from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class SUTIn(BaseModel):
    system_id: str
    name: Optional[str] = None
    endpoint: str
    method: str = "POST"
    headers: Dict[str, Any] = Field(default_factory=dict)
    body: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None

class MappingDetectIn(BaseModel):
    system_id: str
    headers: Dict[str, Any]
    body: Dict[str, Any]

class MappingOut(BaseModel):
    system_id: str
    prompt_paths: List[str]
    response_paths: List[str]
    error_rules: Dict[str, Any]

class PlanIn(BaseModel):
    run_id: str
    system_id: str
    mode: str  # standard | manual
    standard_code: Optional[str] = None
    metric_codes: List[str]
    dataset: Dict[str, Any]  # {type: single|inline|jsonl, value|lines|path}

class RunIn(BaseModel):
    run_id: str

class EvaluateIn(BaseModel):
    run_id: str
    metric_codes: List[str]

class RunResultOut(BaseModel):
    run_id: str
    overall_score: float
    metric_pass_fail: Dict[str, bool]


class DatasetProfile(BaseModel):
    modalities: List[str] = []
    languages: List[str] = []
    domains: List[str] = []
    tags: List[str] = []
    has_ground_truth: bool = False
    item_schema: Dict[str, Any] = Field(default_factory=dict)


class DatasetIn(BaseModel):
    dataset_id: str
    name: str
    description: Optional[str] = None
    source: Optional[str] = None
    license: Optional[str] = None
    profile: DatasetProfile


class DatasetItemIn(BaseModel):
    dataset_id: str
    item_index: int
    payload: Dict[str, Any]
    tags: List[str] = []


class StandardMetricIn(BaseModel):
    standard_code: str
    metric_code: str
    required: bool = False


class MetricUpdateIn(BaseModel):
    code: str
    unit: Optional[str] = None
    aggregation: Optional[str] = None
    applicability: Optional[Dict[str, Any]] = None
