from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base

class System(Base):
    __tablename__ = "systems"
    id = Column(Integer, primary_key=True)
    system_id = Column(String, unique=True, index=True)
    name = Column(String)
    endpoint = Column(String)
    method = Column(String)
    headers = Column(Text)           # JSON string
    body = Column(Text)              # JSON string
    expected_output = Column(Text)   # JSON string
    created_at = Column(String)
    updated_at = Column(String)

class Mapping(Base):
    __tablename__ = "mappings"
    id = Column(Integer, primary_key=True)
    system_id = Column(String, ForeignKey("systems.system_id"))
    prompt_paths = Column(Text)      # JSON array string
    response_paths = Column(Text)    # JSON array string
    error_rules = Column(Text)       # JSON object string
    created_at = Column(String)
    # New fields for placeholder/extraction-driven flow
    input_placeholder = Column(String, nullable=True)  # e.g., "${input}" (default)
    message_extractor = Column(Text, nullable=True)    # Python function code as text
    session_id_field = Column(String, nullable=True)   # top-level key name for session id
    api_key_name = Column(String, nullable=True)       # header key for API key
    api_key_value = Column(String, nullable=True)      # header value for API key

class Standard(Base):
    __tablename__ = "standards"
    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True)
    name = Column(String)
    description = Column(Text)
    version = Column(String, nullable=True)

class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True)
    standard_code = Column(String, ForeignKey("standards.code"), nullable=True)
    description = Column(Text)
    type = Column(String)            # function | llm_scoring
    weight = Column(Float)
    threshold = Column(Float)
    unit = Column(String, nullable=True)
    aggregation = Column(String, nullable=True)
    applicability = Column(Text, nullable=True)   # JSON
    config = Column(Text)            # JSON

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, unique=True, index=True)
    system_id = Column(String, ForeignKey("systems.system_id"))
    evaluation_id = Column(String, nullable=True)
    mode = Column(String)            # standard | manual
    standard_code = Column(String, ForeignKey("standards.code"), nullable=True)
    dataset_ref = Column(String)
    started_at = Column(String)
    finished_at = Column(String)
    status = Column(String)
    overall_score = Column(Float)
    mapping_snapshot = Column(Text, nullable=True)
    selected_metric_codes = Column(Text, nullable=True)   # JSON array string

class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("runs.run_id"))
    idx = Column(Integer)
    request = Column(Text)           # JSON
    response = Column(Text)          # JSON
    latency_ms = Column(Integer)
    error_detected = Column(Integer) # 0/1

class MetricResult(Base):
    __tablename__ = "metric_results"
    id = Column(Integer, primary_key=True)
    run_id = Column(String, ForeignKey("runs.run_id"))
    metric_code = Column(String, ForeignKey("metrics.code"))
    result = Column(Text)            # JSON
    passed = Column(Integer)         # 0/1


class StandardMetric(Base):
    __tablename__ = "standards_metrics"
    id = Column(Integer, primary_key=True)
    standard_code = Column(String, ForeignKey("standards.code"))
    metric_code = Column(String, ForeignKey("metrics.code"))
    required = Column(Integer, default=0)  # 0/1


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(String, unique=True, index=True)
    name = Column(String)
    description = Column(Text)
    source = Column(String)
    license = Column(String)
    profile = Column(Text)     # JSON
    created_at = Column(String)


class DatasetItem(Base):
    __tablename__ = "dataset_items"
    id = Column(Integer, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.dataset_id"))
    item_index = Column(Integer)
    payload = Column(Text)     # JSON per item
    tags = Column(Text)        # JSON array
