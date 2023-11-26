class Simulation:

    def __init__(self) -> None:
        pass

    def run(self):
        return NotImplementedError

    def init_params(self):
        return NotImplementedError

    def update(self):
        return NotImplementedError