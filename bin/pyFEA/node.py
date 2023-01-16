
class Node:
    def __init__(self, id, x, y, z, label=None):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.label = label

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self):
        
