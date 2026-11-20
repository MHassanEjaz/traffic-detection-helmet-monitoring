
# import uuid
# from pathlib import Path

# from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
# from fastapi.responses import FileResponse

# from app.core.config import HELMET_MODELS
# from app.core.video_processor import process_video


# router = APIRouter()

# # Project ke andar API uploads aur processed videos store honge.
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# JOBS_DIR = PROJECT_ROOT / "storage" / "api_jobs"
# JOBS_DIR.mkdir(parents=True, exist_ok=True)

# # Initial prototype: job status memory mein store hoga.
# jobs = {}

# ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# def run_processing_job(
#     job_id: str,
#     input_path: Path,
#     output_path: Path,
#     helmet_model_path: Path
# ):
#     """Background mein video process karta hai."""

#     jobs[job_id]["status"] = "processing"
#     jobs[job_id]["message"] = "Video processing started."

#     def update_progress(progress, message):
#         jobs[job_id]["progress"] = round(progress * 100, 2)
#         jobs[job_id]["message"] = message

#     try:
#         counts, processed_frames = process_video(
#             input_path=input_path,
#             output_path=output_path,
#             helmet_model_path=helmet_model_path,
#             progress_callback=update_progress
#         )

#         jobs[job_id].update({
#             "status": "completed",
#             "progress": 100,
#             "message": "Video processing completed.",
#             "counts": counts,
#             "processed_frames": processed_frames,
#             "output_path": str(output_path)
#         })

#     except Exception as error:
#         jobs[job_id].update({
#             "status": "failed",
#             "message": "Video processing failed.",
#             "error": str(error)
#         })


# @router.post("/process")
# async def upload_and_process_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     helmet_model: str = Form("RideSafe Helmet Model")
# ):
#     """Video upload karta hai aur background processing start karta hai."""

#     extension = Path(file.filename or "").suffix.lower()

#     if extension not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail="Unsupported video format."
#         )

#     if helmet_model not in HELMET_MODELS:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid helmet model selection."
#         )

#     job_id = str(uuid.uuid4())
#     job_dir = JOBS_DIR / job_id
#     job_dir.mkdir(parents=True, exist_ok=True)

#     input_path = job_dir / f"input{extension}"
#     output_path = job_dir / "processed.mp4"

#     try:
#         with input_path.open("wb") as destination:
#             while chunk := await file.read(1024 * 1024):
#                 destination.write(chunk)
#     except Exception:
#         raise HTTPException(
#             status_code=500,
#             detail="Could not save uploaded video."
#         )
#     finally:
#         await file.close()

#     jobs[job_id] = {
#         "status": "queued",
#         "progress": 0,
#         "message": "Video uploaded. Waiting for processing.",
#         "counts": None,
#         "processed_frames": None,
#         "error": None
#     }

#     background_tasks.add_task(
#         run_processing_job,
#         job_id,
#         input_path,
#         output_path,
#         Path(HELMET_MODELS[helmet_model])
#     )

#     return {
#         "job_id": job_id,
#         "status": "queued",
#         "message": "Video uploaded successfully. Processing has been queued."
#     }


# @router.get("/jobs/{job_id}/status")
# def get_job_status(job_id: str):
#     """Processing progress aur status return karta hai."""

#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job not found.")

#     job = jobs[job_id]

#     return {
#         "job_id": job_id,
#         "status": job["status"],
#         "progress": job["progress"],
#         "message": job["message"],
#         "error": job["error"]
#     }


# @router.get("/jobs/{job_id}/result")
# def get_job_result(job_id: str):
#     """Completed job ke counts aur video link return karta hai."""

#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job not found.")

#     job = jobs[job_id]

#     if job["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail=f"Job is not completed. Current status: {job['status']}"
#         )

#     return {
#         "job_id": job_id,
#         "status": job["status"],
#         "processed_frames": job["processed_frames"],
#         "counts": job["counts"],
#         "video_url": f"/api/jobs/{job_id}/video"
#     }


# @router.get("/jobs/{job_id}/video")
# def download_processed_video(job_id: str):
#     """Processed MP4 video download karne ke liye."""

#     if job_id not in jobs:
#         raise HTTPException(status_code=404, detail="Job not found.")

#     job = jobs[job_id]

#     if job["status"] != "completed":
#         raise HTTPException(
#             status_code=409,
#             detail="Processed video is not ready yet."
#         )

#     output_path = Path(job["output_path"])

#     if not output_path.exists():
#         raise HTTPException(
#             status_code=404,
#             detail="Processed video file not found."
#         )

#     return FileResponse(
#         path=str(output_path),
#         media_type="video/mp4",
#         filename="processed_traffic_video.mp4"
#     )




import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import HELMET_MODELS
from app.core.video_processor import process_video
from app.db.database import SessionLocal, get_db
from app.db.models import VideoJob


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = PROJECT_ROOT / "storage" / "api_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def run_processing_job(
    job_id: str,
    input_path: Path,
    output_path: Path,
    helmet_model_path: Path,
):
    """Background mein video process aur database update karta hai."""

    db = SessionLocal()

    try:
        job = db.query(VideoJob).filter(
            VideoJob.job_id == job_id
        ).first()

        if job is None:
            return

        job.status = "processing"
        job.message = "Video processing started."
        db.commit()

        last_saved_progress = -1

        def update_progress(progress, message):
            nonlocal last_saved_progress

            progress_percent = round(float(progress) * 100, 2)

            # Database ko har frame par update nahi karna.
            # Har 1% progress par update save hoga.
            if (
                progress_percent >= last_saved_progress + 1
                or progress_percent >= 100
            ):
                job.progress = progress_percent
                job.message = message
                db.commit()

                last_saved_progress = progress_percent

        counts, processed_frames = process_video(
            input_path=input_path,
            output_path=output_path,
            helmet_model_path=helmet_model_path,
            progress_callback=update_progress,
        )

        job.status = "completed"
        job.progress = 100
        job.message = "Video processing completed."
        job.counts = counts
        job.processed_frames = processed_frames
        job.output_path = str(output_path)

        db.commit()

    except Exception as error:
        db.rollback()

        job = db.query(VideoJob).filter(
            VideoJob.job_id == job_id
        ).first()

        if job is not None:
            job.status = "failed"
            job.message = "Video processing failed."
            job.error = str(error)
            db.commit()

    finally:
        db.close()


@router.post("/process")
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    helmet_model: str = Form("RideSafe Helmet Model"),
    db: Session = Depends(get_db),
):
    """Video upload karta hai aur background processing start karta hai."""

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    if helmet_model not in HELMET_MODELS:
        raise HTTPException(
            status_code=400,
            detail="Invalid helmet model selection.",
        )

    job_id = str(uuid.uuid4())

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / f"input{extension}"
    output_path = job_dir / "processed.mp4"

    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                destination.write(chunk)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded video.",
        )

    finally:
        await file.close()

    # Job record database mein save karo.
    new_job = VideoJob(
        job_id=job_id,
        status="queued",
        progress=0,
        message="Video uploaded. Waiting for processing.",
        input_filename=Path(file.filename or "video").name,
        helmet_model=helmet_model,
        input_path=str(input_path),
        output_path=str(output_path),
        counts=None,
        processed_frames=None,
        error=None,
    )

    try:
        db.add(new_job)
        db.commit()

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Could not save job to database.",
        )

    background_tasks.add_task(
        run_processing_job,
        job_id,
        input_path,
        output_path,
        Path(HELMET_MODELS[helmet_model]),
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded successfully. Processing has been queued.",
    }


@router.get("/jobs/{job_id}/status")
def get_job_status(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Database se processing progress aur status return karta hai."""

    job = db.query(VideoJob).filter(
        VideoJob.job_id == job_id
    ).first()

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
    }


@router.get("/jobs/{job_id}/result")
def get_job_result(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Completed job ke counts aur video link return karta hai."""

    job = db.query(VideoJob).filter(
        VideoJob.job_id == job_id
    ).first()

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed. Current status: {job.status}",
        )

    return {
        "job_id": job.job_id,
        "status": job.status,
        "processed_frames": job.processed_frames,
        "counts": job.counts,
        "video_url": f"/api/jobs/{job_id}/video",
    }


@router.get("/jobs/{job_id}/video")
def download_processed_video(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Processed MP4 video download karne ke liye."""

    job = db.query(VideoJob).filter(
        VideoJob.job_id == job_id
    ).first()

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="Processed video is not ready yet.",
        )

    output_path = Path(job.output_path)

    if not output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video file not found.",
        )

    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename="processed_traffic_video.mp4",
    )
