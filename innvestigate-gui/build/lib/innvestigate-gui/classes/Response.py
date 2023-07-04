class Response:
    def __init__(self, code, status, message=None, content=None):
        self.code = code
        self.status = status
        self.message = message
        self.content = content
    
    def toJSON(self):
        return self.__dict__
