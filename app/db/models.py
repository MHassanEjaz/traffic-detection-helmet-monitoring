
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Text,
    DateTime,
    JSON,
    func,
)

from app.db.database import Base


class VideoJob(Base):
    __tablename__ = "jobs"

    # Unique job identifier
    job_id = Column(String(36), primary_key=True, index=True)

    # Job information
    status = Column(String(20), nullable=False, default="queued")
    progress = Column(Float, nullable=False, default=0.0)
    message = Column(Text, nullable=True)

    # Video information
    input_filename = Column(String(255), nullable=True)
    helmet_model = Column(String(100), nullable=True)
    input_path = Column(Text, nullable=True)
    output_path = Column(Text, nullable=True)

    # Processing results
    counts = Column(JSON, nullable=True)
    processed_frames = Column(Integer, nullable=True)

    # Error information
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
