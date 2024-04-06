'''
MAIN IDEA TO FIND TERRITORY:
    - for each empty point (four directions):
        - empty points + edges = 4
            waiting_list
        - four directions have different colour stones
            free list
        - same colour stones + edges = 4
            territory list
        - same colour stones + edges + empty points = 4
            ~ for empty points:
                * if one in the free list:
                    free list
                    if other empty points in waiting list: 
                        add the belongs group and delete it in the waiting list
                * if one in the territory list:
                    territory list
                    waiting list check
                * if one in the waiting list:compare the colour
                    different colour: move all points to the free list
                    same colour or waiting_list[0] is None: add empty point in the waiting list 
                * not included in any list:
                    only add the parent empty point in the waiting list
'''

class GoScore():
    '''
    Calculate the score for a given board
    - Use four list to claculate the territory for each player
        * waiting_list: store the points that has no group
        * black_territoy_list: black player's territory points
        * white_territoy_list: white player's territory points
        * free_list: Free points 
    '''
    def __init__(self, board, waiting_list=[], black_territory_list=[],white_territory_list=[],free_list=[]):
        self.board = board
        self.waiting_list = waiting_list
        self.black_territory_list = black_territory_list
        self.white_territory_list = white_territory_list
        self.free_list = free_list
        
    def sort_list(self, point, empty, color):
        '''
        sort the points
        '''
        # if one empty point in the free list, then all the other 
        # empty points and the current point are free
        is_free = False
        is_territory = False
        is_waiting = False
        waiting_child = []
        free_list = self.free_list
        waiting_list = self.waiting_list
        black_territory_list = self.black_territory_list
        white_territory_list = self.white_territory_list
        for e in empty:
            if e in self.free_list:
                is_free = True
                break
            if (color == 'w' and e in white_territory_list) or \
                (color == 'b' and e in black_territory_list):
                is_territory = True
                break
            for w in waiting_list:
                if e in w[1]:
                    if w[1] not in waiting_child:
                        waiting_child.append(w)
                    is_waiting = True
                    
        if is_free:       
            empty.append(point)
            free_list = add_to_list(free_list,empty)
            # check if there is point still in waiting_list
            # if in the waiting list, return the list of points and add all the points in the free list
            need_to_add, waiting_list = in_waiting_list(empty, waiting_list)
            if len(need_to_add) > 0:
                free_list = add_to_list(free_list, need_to_add)
            
        elif is_territory:
            if color == 'w':
                if point not in white_territory_list:
                    white_territory_list.append(point)
                white_territory_list = add_to_list(white_territory_list, empty)
                # check if there is point still in waiting _list
                empty.append(point)
                need_to_add, waiting_list= in_waiting_list(empty, waiting_list)
                if len(need_to_add) > 0:
                    white_territory_list = add_to_list(white_territory_list, need_to_add)
            elif color == 'b':
                if point not in black_territory_list:
                    black_territory_list.append(point)
                black_territory_list = add_to_list(black_territory_list, empty)
                # check if there is point still in waiting _list
                empty.append(point)
                need_to_add, waiting_list= in_waiting_list(empty, waiting_list)
                if len(need_to_add) > 0:
                    black_territory_list = add_to_list(black_territory_list, need_to_add)
        elif is_waiting: 
            new_waiting = [point]
            new_free = [point]
            diffrent_color = False
            # compare the color and the waiting list color
            for child in waiting_child:
                if color == child[0] or child[0] == None:
                    new_waiting = add_to_list(new_waiting,child[1:][0])
                    if child in waiting_list:
                        waiting_list.remove(child)
                elif color != child[0]:
                    diffrent_color = True
                    new_free = add_to_list(new_free, child[1:][0])
                    if child in waiting_list:
                        waiting_list.remove(child)
            if diffrent_color:
                combined = new_free.copy()
                for w in new_waiting:
                    if w not in combined:
                        combined.append(w)
                free_list = add_to_list(free_list, combined)
            else:
                new_list = [color,new_waiting]
                waiting_list.append(new_list)
                        
        else:
            waiting_list.append([color,[point]])
        board = self.board
        return GoScore(board,waiting_list,black_territory_list,white_territory_list,free_list)                

    def process_waiting_list(self):
        '''
        Recheck waiting list and free list
        '''
        new_waiting_list = []
        # print(waiting_list)
        
        for i in range(len(self.waiting_list)-1):
            if self.waiting_list[i] != None:
                exist = False
                color_group = [self.waiting_list[i][0]]
                point_group = []
                
                for point in self.waiting_list[i][1]:
                    point_group.append(point)
                
                neighbors = find_neighbour(point_group)
                
                for j in range(i+1,len(self.waiting_list)):
                    if self.waiting_list[j]:
                        for point in self.waiting_list[j][1]:
                            if point in neighbors:
                                exist = True
                                color_group.append(self.waiting_list[j][0])
                                for p in self.waiting_list[j][1]:
                                    point_group.append(p)
                            if exist:
                                self.waiting_list[j] = None
                                break
                if not exist:
                    new_waiting_list.append(self.waiting_list[i])
                else:
                    color = get_colour(color_group)
                    if color == False:
                        for point in point_group:
                            add_to_list(self.free_list,point)
                    else:
                        if color == 'white':
                            new_list = ['white', point_group]
                        elif color == "black":
                            new_list = ['black', point_group]
                        else:
                            new_list = [None, point_group]
                        new_waiting_list.append(new_list)
        if len(self.waiting_list) != 0 and self.waiting_list[-1] != None:
            new_waiting_list.append(self.waiting_list[-1])    
                
        self.waiting_list = new_waiting_list.copy()

    def territory_waiting_list(self, white_stones, black_stones):
        '''
        recheck the waiting list and territory list
        '''
        new_waiting_list = self.waiting_list.copy()
        for wlist in new_waiting_list:
            if wlist[0] == 'white':
                is_territory = True
                neighbors = find_neighbour(wlist[1])
                for neighbor in neighbors:
                    if neighbor not in white_stones:
                        is_territory = False
                if is_territory:
                    self.waiting_list.remove(wlist)
                    for w in wlist[1]:
                        self.white_territory_list.append(w)
                    
            if wlist[0] == 'black':
                is_territory = True
                neighbors = find_neighbour(wlist[1])
                print(neighbors)
                for neighbor in neighbors:
                    if neighbor not in black_stones:
                        is_territory = False
                if is_territory:
                    self.waiting_list.remove(wlist)
                    for w in wlist[1]:
                        self.black_territory_list.append(w)

    def go_score(self, white_stones, black_stones):
        '''
        calculate the final score
        '''
        blacks = len(black_stones) + len(self.black_territory_list)
        whites = len(white_stones) + len(self.white_territory_list)
        public_score = 81 - blacks - whites
        black_score = blacks + (public_score/2) - (6.5/2)
        white_score = whites + (public_score/2) + (6.5/2)
        
        return black_score, white_score
    
    # **main function**
    def territory(self):
        '''
        find black and white territories
        '''
        black_stones,white_stones, empty_list= group_stones(self.board)
        
        for point in empty_list:
            # find four directions
            four_directions = [[point[0]-1, point[1]],[point[0]+1, point[1]], [point[0], point[1]-1],[point[0],point[1]+1]] # [up, down, left, right]
            
            black,white,edge,empty = [],[],[],[]

            for direction in four_directions:
                if direction in white_stones:
                    white.append(direction)
                elif direction in black_stones:
                    black.append(direction)
                elif (direction[0]<0 or direction[1] < 0 or direction[0] > 8 or direction[1] > 8):
                    edge.append(direction)
                else:
                    empty.append(direction)

            if len(empty) + len(edge) == 4: # empty points + edges = 4
                self.waiting_list.append([None,[point]])
            elif len(white) > 0 and len(black) > 0: # with different color stones
                if point not in self.free_list:
                    self.free_list.append(point)
            elif len(white) + len(edge) == 4: # white color stones + edges = 4
                self.white_territory_list.append(point)
            elif len(black) + len(edge) == 4: # black color stones + edges = 4
                self.black_territory_list.append(point)
            elif len(white) + len(edge) + len(empty) == 4 and len(empty) != 0:
                # white
                self.sort_list(point, empty,'w')
            elif len(black) + len(edge) + len(empty) == 4 and len(empty) != 0:
                # black
                self.sort_list(point, empty,'b')
        black_score, white_scores = self.go_score(white_stones,black_stones)
        
        return black_score, white_scores             



def get_colour(colours):
    '''
    check if the list of colours are the same:
        - All white: return 'white'
        - All black: return 'black'
        - Others: return False
    '''        
    if 'black' in colours and 'white' not in colours:
        return 'black'
    elif 'white' in colours and 'black' not in colours:
        return 'white'
    else:
        return False

def find_neighbour(points):
    '''
    The neighbours of the point
    '''
    neightbours = []
    for point in points:
        if point[0]-1>=0 and [point[0]-1, point[1]] not in points:
            neightbours.append([point[0]-1, point[1]])
        if point[0]+1<=8 and [point[0]+1, point[1]] not in points:
            neightbours.append([point[0]+1, point[1]])
        if point[1]-1>=0 and [point[0], point[1]-1] not in points:
            neightbours.append([point[0], point[1]-1])
        if point[1]+1<=8 and [point[0], point[1]+1] not in points:
            neightbours.append([point[0], point[1]+1])
    return neightbours

def group_stones(board):
    '''
    Classify the white stones, black stones and empty points in the board
    '''
    black_stones, white_stones, empty_points = [],[],[]
    
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0: # empty
                empty_points.append([row, col])
            elif board[row][col] == 1: # black
                black_stones.append([row, col])
            else: # white
                white_stones.append([row, col])
                
    return black_stones, white_stones, empty_points

def check_in_groups(groups, stone):
    '''
    check if the stone is already in a group
        - yes: return index
        - no: return fasle
    '''
    for i in range(len(groups)):
        if stone in groups[i]:
            return i
    return False

def find_groups(stones_list, stone_groups):
    '''
    For the same colour stones that are adjacent to each other, group them together
    '''
    for stone in stones_list:
        # find if there are same color stones at eight directions
        eight_directions = [[stone[0]-1,stone[1]-1],[stone[0]-1,stone[1]+1], # top left, top right
                            [stone[0]+1,stone[1]-1],[stone[0]+1, stone[1]+1], # bottom left, bottom right
                            [stone[0]-1,stone[1]],[stone[0]+1, stone[1]], # top, bottom
                            [stone[0],stone[1]-1],[stone[0], stone[1]+1] # left, right
                            ]
        for adjacent in eight_directions:
            if adjacent in stones_list:
                stone_check = check_in_groups(stone_groups, stone)
                
                # if the current stone already in a group, then add the adjacent stone in this group
                if type(stone_check) == int :
                    if adjacent in stone_groups and adjacent not in stone_groups[stone_check]:
                        stone_groups[stone_check].append(adjacent)
                # else check if the adjacent stone already in a group
                # if yes, add the current stone in this group
                # else, create a new group and add it in the groups
                else:
                    adjacent_check = check_in_groups(stone_groups, adjacent)
                    if type(adjacent_check) == int and stone not in stone_groups[adjacent_check]: # this stone is already in a group
                        stone_groups[adjacent_check].append(stone)
                    else: # create a new group
                        stone_groups.append([stone, adjacent])
    return stone_groups

def in_waiting_list(point_list, waiting_list):
    '''
    - find the points in the waiting list that need to be free
    - remove the free points in the waiting list
    '''
    need_to_add = []
    for point in point_list:
        for waiting in waiting_list:
            if point in waiting[1]:
                add_to_list(need_to_add, waiting[1])
                # need_to_add.append(w for w in waiting[1] if w != point)
                waiting_list.remove(waiting)
    return need_to_add, waiting_list

def add_to_list(list1, list2):
    '''
    group two list
    '''
    for l in list2:
        if l not in list1:
            list1.append(l)
    return list1

