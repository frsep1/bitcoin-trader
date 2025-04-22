import data_reader
import numpy as np
import datetime


start = data_reader.time.mktime(datetime.datetime.strptime(data_reader.start, "%d/%m/%Y").timetuple())
end = data_reader.time.mktime(datetime.datetime.strptime(data_reader.end, "%d/%m/%Y").timetuple())
data = data_reader.historic_data(start=data_reader.start, end=data_reader.end)

class whale():
    def __init__(self, scoring, dimentions, minx, maxx, start, end, data):
        self.scoring = scoring
        self.pos = np.random.uniform(minx, maxx, size=dimentions)
        self.score = scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], start, end, my_data=data)
    def change_pos(self, new_pos):
        self.pos = new_pos
        self.score = self.scoring(self.pos[0:6], self.pos[6:12], self.pos[12:14], start, end, my_data=data)
        
def WOA(scoring, dimentions, minx, maxx, num_whales, iterations, spiral_constant=1, start=start, end=end, data=data):
    whales = [whale(scoring, dimentions, minx, maxx, start, end, data) for i in range(num_whales)]
    best_pos = np.zeros(dimentions)
    best_score = 0
    for i in range(num_whales):
        if whales[i].score > best_score:
            best_score = whales[i].score
            best_pos = whales[i].pos
    for i in range(iterations):
        a = 2 * (1 - i / iterations)
        a2 = -1 + i * (-1 / iterations)
        for j in range(num_whales):
            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = (2 * np.random.rand()) - 1
            b = spiral_constant
            p = np.random.rand()
            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_pos - whales[j].pos)
                    new_pos = best_pos - A * D
                    whales[j].change_pos(new_pos)

                else:
                    random_whale = whales[np.random.randint(0, num_whales - 1)]
                    D = abs(C * random_whale.pos - whales[j].pos)
                    new_pos = random_whale.pos - A * D
                    whales[j].change_pos(new_pos)
            else:
                d = abs(best_pos - whales[j].pos)
                new_pos = d * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
                whales[j].change_pos(new_pos)
        for j in range(num_whales):
            if whales[j].score > best_score:
                best_score = whales[j].score
                best_pos = whales[j].pos
    return best_pos, best_score

best_pos, best_score = WOA(data_reader.scoring, 14, 1, 100, 10, 10, start=start, end=end, data=data)
print("Best position: ", best_pos)
print("Best score: ", best_score)


        
                

