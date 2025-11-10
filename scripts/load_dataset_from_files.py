#!/usr/bin/env python3
import sys, json, datetime
from pathlib import Path

from backend.db import SessionLocal, engine, Base
from backend.models import Dataset, DatasetItem


def now_iso():
    import datetime
    return datetime.datetime.utcnow().isoformat()


def main(profile_path: str, items_path: str):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        profile = json.loads(Path(profile_path).read_text())
        dsid = profile.get("dataset_id")
        if not dsid:
            raise SystemExit("profile JSON missing dataset_id")
        ds = db.query(Dataset).filter_by(dataset_id=dsid).first()
        data = dict(
            dataset_id=dsid,
            name=profile.get("name", dsid),
            description=profile.get("description", ""),
            source=profile.get("source", ""),
            license=profile.get("license", ""),
            profile=json.dumps(profile.get("profile", {})),
            created_at=now_iso(),
        )
        if ds:
            for k, v in data.items():
                setattr(ds, k, v)
        else:
            ds = Dataset(**data)
            db.add(ds)
        # load items
        idx = 1
        with open(items_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                row = db.query(DatasetItem).filter_by(dataset_id=dsid, item_index=idx).first()
                if row:
                    row.payload = json.dumps(payload)
                    row.tags = json.dumps([])
                else:
                    row = DatasetItem(dataset_id=dsid, item_index=idx, payload=json.dumps(payload), tags=json.dumps([]))
                    db.add(row)
                idx += 1
        db.commit()
        print(f"Loaded dataset '{dsid}' with {idx-1} items.")
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/load_dataset_from_files.py <profile.json> <items.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

