"""List registered PRCF tasks."""


def main():
    import prcf.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks
    for task in list_tasks():
        if task.startswith('PRCF-'):
            print(task)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
