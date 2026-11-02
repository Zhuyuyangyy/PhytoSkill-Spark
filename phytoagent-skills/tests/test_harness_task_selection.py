"""Pure tests for A/B CLI task parsing and task-set selection."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from harness.task_selection import add_ab_task_selection_arguments, build_ab_tasks


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_ab_task_selection_arguments(parser)
    return parser


def test_real_image_kind_filter_is_applied_to_real_tasks_after_selection():
    fixture_tasks = [Task("fixture:negative", "negative_trigger")]
    real_tasks = [
        Task("real:herb", "end_to_end"),
        Task("real:leaf", "end_to_end"),
        Task("real:negative", "negative_trigger"),
    ]
    builder_calls = []

    def build_tasks(*, only_real_images=False):
        builder_calls.append(only_real_images)
        return real_tasks if only_real_images else fixture_tasks

    args = _parser().parse_args(
        ["--real-images", "--kind", "negative_trigger"]
    )
    tasks = build_ab_tasks(
        build_tasks, kinds=args.kind, real_images=args.real_images
    )

    assert builder_calls == [True]
    assert [task.task_id for task in tasks] == ["real:negative"]
    assert all(task.kind == "negative_trigger" for task in tasks)


def test_real_images_without_kind_keeps_the_complete_real_task_set():
    real_tasks = [
        Task("real:herb", "end_to_end"),
        Task("real:leaf", "end_to_end"),
        Task("real:negative", "negative_trigger"),
    ]

    def build_tasks(*, only_real_images=False):
        return real_tasks if only_real_images else []

    args = _parser().parse_args(["--real-images"])
    tasks = build_ab_tasks(
        build_tasks, kinds=args.kind, real_images=args.real_images
    )

    assert [task.task_id for task in tasks] == [
        "real:herb", "real:leaf", "real:negative"
    ]


def test_kind_only_filters_the_fixture_task_set():
    fixture_tasks = [
        Task("fixture:positive", "positive_trigger"),
        Task("fixture:negative", "negative_trigger"),
    ]
    builder_calls = []

    def build_tasks(*, only_real_images=False):
        builder_calls.append(only_real_images)
        return fixture_tasks

    args = _parser().parse_args(["--kind", "negative_trigger"])
    tasks = build_ab_tasks(
        build_tasks, kinds=args.kind, real_images=args.real_images
    )

    assert builder_calls == [False]
    assert [task.task_id for task in tasks] == ["fixture:negative"]
