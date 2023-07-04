class ClassificationMapping:
    def __init__(self, classId, className, score):
        self.classId = classId
        self.className = className
        self.score = score

    def toJSON(self):
        return self.__dict__
