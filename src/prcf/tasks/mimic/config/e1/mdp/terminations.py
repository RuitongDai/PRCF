"""Reference completion is an explicit episode boundary."""


def motion_finished(env, command_name="motion"):
    return env.command_manager.get_term(command_name).finished
