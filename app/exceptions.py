class InvalidFacePoseError(Exception):

    # self passes InvalidFacePoseError into the function below
    def __init__(self, message:str, yaw, pitch, max_yaw, max_pitch,):

        # super gives access to parent class - Exception
        # this is essentially Exception.__init__(self, message)
        super().__init__(message)

        # then save all these values inside exception object
        self.message = message
        self.yaw = yaw
        self.pitch = pitch
        self.max_yaw = max_yaw
        self.max_pitch = max_pitch