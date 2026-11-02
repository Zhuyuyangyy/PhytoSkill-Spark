"""Pure CLI task selection for the A/B harness.

This module intentionally depends only on the standard library so argument
parsing and task-set selection can be tested without loading a model backend.
"""


def add_ab_task_selection_arguments(parser):
    """Register task selection options shared by the A/B CLI and its tests."""
    parser.add_argument("--kind", action="append", help="restrict to a task kind")
    parser.add_argument(
        "--real-images",
        action="store_true",
        help=(
            "run only the real-photograph tasks. Each carries its own mode "
            "(live/herb) and needs the images on disk plus a reachable DGX "
            "Spark node. Not mixable with the fixture set, whose inputs every "
            "real mode refuses by design."
        ),
    )


def build_ab_tasks(task_builder, *, kinds=None, real_images=False):
    """Build the requested A/B task set, then apply any kind restriction."""
    tasks = (task_builder(only_real_images=True) if real_images
             else task_builder())
    if kinds:
        tasks = [task for task in tasks if task.kind in kinds]
    return tasks
