from datetime import datetime

from pydantic import BaseModel, field_validator, model_validator

from data.local_data import get_regions

ALLOWED_REGIONS = get_regions()


class IngestValidator(BaseModel):
    region: str
    start_date: str
    end_date: str

    @field_validator("region")
    def validate_region(cls, v):
        if v not in ALLOWED_REGIONS:
            raise ValueError(f"Region must be one of {ALLOWED_REGIONS}")
        return v

    @field_validator("start_date", "end_date")
    def validate_date_format(cls, v):
        try:
            # YY-MM-DD
            return datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Date must be in format 'YY-MM-DD'")

    @model_validator(mode="after")
    def check_date_order(self):
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be later than start_date")
        return self
