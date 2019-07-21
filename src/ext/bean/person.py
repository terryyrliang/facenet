## @author Terry LIANG
##

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Person:
    def __init__(self, person_id, person_pos_lt, person_pos_rb):
        self.person_id = person_id
        self.person_pos_lt = person_pos_lt
        self.person_pos_rb = person_pos_rb
        self.name = 'unknown'


detect_storage = {}
person_storage = {}

# a = person(1, position(1, 2), position(5, 6))
# person_storage[1] = a
# print(person_storage[1].person_pos_lt.x)