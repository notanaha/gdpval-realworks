"""Tests for RepoBootstrapper snapshot validation.

Usage:
    pytest tests/test_repo_bootstrapper.py -v
"""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from core.config import EXPECTED_TASK_COUNT

# Skip all tests if huggingface_hub is not available
pytest.importorskip("huggingface_hub")

import core.repo_bootstrapper as repo_bootstrapper


def _make_rows(start: int, count: int, *, with_files: bool = False) -> list[dict]:
    """Build sequential test rows for parquet shards.

    Args:
        start: Starting numeric suffix for generated task IDs.
        count: Number of rows to generate.
        with_files: Whether deliverable_files should contain one file per row.
    """
    rows = []
    for index in range(start, start + count):
        task_id = f"task_{index:03d}"
        rows.append(
            {
                "task_id": task_id,
                "prompt": f"Prompt {index}",
                "sector": "Finance and Insurance",
                "occupation": "Analyst",
                "rubric_json": "{}",
                "rubric_pretty": "{}",
                "deliverable_text": "",
                "deliverable_files": [f"{task_id}.txt"] if with_files else [],
            }
        )
    return rows


def _write_snapshot(root: Path, shard_sizes: list[int], *, with_files: bool = False) -> list[str]:
    """Create a sharded snapshot layout and return the generated task IDs."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (root / "reference_files").mkdir(parents=True, exist_ok=True)

    task_ids: list[str] = []
    start = 0
    total_shards = len(shard_sizes)
    for shard_index, shard_size in enumerate(shard_sizes):
        rows = _make_rows(start, shard_size, with_files=with_files)
        task_ids.extend(row["task_id"] for row in rows)
        df = pd.DataFrame(rows)
        df.to_parquet(
            data_dir / f"train-{shard_index:05d}-of-{total_shards:05d}.parquet",
            index=False,
        )
        start += shard_size

    return task_ids


def _write_manifest(workspace_dir: Path, task_ids: list[str], *, needs_files: bool = False) -> None:
    """Write a minimal step0_needs_files_manifest.json for the given tasks."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tasks": {
            task_id: {
                "needs_files": needs_files,
                "original_file_count": 1 if needs_files else 0,
                "original_files": [f"{task_id}.txt"] if needs_files else [],
            }
            for task_id in task_ids
        }
    }
    (workspace_dir / "step0_needs_files_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


class TestRepoBootstrapperValidation:
    @patch("core.repo_bootstrapper.HfApi")
    def test_validate_snapshot_uses_sample_size_across_multiple_shards(
        self, mock_hf_api_class, tmp_path, monkeypatch
    ):
        """Smoke-test bootstrap validation should honor sample_size across shards."""
        mock_hf_api_class.return_value = object()
        local_path = tmp_path / "snapshot"
        task_ids = _write_snapshot(local_path, [2, 1])
        workspace_dir = tmp_path / "workspace"
        _write_manifest(workspace_dir, task_ids)
        monkeypatch.setattr(repo_bootstrapper, "WORKSPACE_DIR", workspace_dir)

        bootstrapper = repo_bootstrapper.RepoBootstrapper(
            submission_repo_id="user/exp998",
            local_path=str(local_path),
            token="hf_test_token",
            expected_rows=3,
        )

        bootstrapper._validate_snapshot()

    @patch("core.repo_bootstrapper.HfApi")
    def test_validate_snapshot_defaults_to_full_dataset_count(
        self, mock_hf_api_class, tmp_path, monkeypatch
    ):
        """Standard bootstrap validation should still default to the full dataset size."""
        mock_hf_api_class.return_value = object()
        local_path = tmp_path / "snapshot"
        task_ids = _write_snapshot(local_path, [110, 110])
        workspace_dir = tmp_path / "workspace"
        _write_manifest(workspace_dir, task_ids)
        monkeypatch.setattr(repo_bootstrapper, "WORKSPACE_DIR", workspace_dir)

        bootstrapper = repo_bootstrapper.RepoBootstrapper(
            submission_repo_id="user/full",
            local_path=str(local_path),
            token="hf_test_token",
        )

        assert bootstrapper.expected_rows == EXPECTED_TASK_COUNT
        bootstrapper._validate_snapshot()


class TestValidatePreUpload:
    def test_validate_pre_upload_counts_rows_across_multiple_shards(self, tmp_path, monkeypatch):
        """Pre-upload validation should aggregate all parquet shards before counting rows."""
        local_path = tmp_path / "snapshot"
        task_ids = _write_snapshot(local_path, [2, 1], with_files=True)
        workspace_dir = tmp_path / "workspace"
        _write_manifest(workspace_dir, task_ids, needs_files=True)
        monkeypatch.setattr(repo_bootstrapper, "WORKSPACE_DIR", workspace_dir)

        errors = repo_bootstrapper.validate_pre_upload(
            local_path=str(local_path),
            expected_rows=3,
        )

        assert errors == []
