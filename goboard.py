import pygame

'''
the borad is represented as a 9*9 matrix:
0 - empty
1 - black player
2 - white player

the coordinates of the borad are represented by 9 characters:
   a  b  c  d  e  f  g  h  i
9  0  0  0  0  0  0  0  0  0
8  0  0  0  0  0  0  0  0  0
7  0  0  0  0  0  0  0  0  0
6  0  0  0  0  0  0  0  0  0
5  0  0  0  0  0  0  0  0  0
4  0  0  0  0  0  0  0  0  0
3  0  0  0  0  0  0  0  0  0
2  0  0  0  0  0  0  0  0  0
1  0  0  0  0  0  0  0  0  0
'''

def draw(num_games):
    '''
    Draw the empty board
    '''
    pygame.init()
    
    # set colors
    white, black, red, brown = (255, 255, 255), (0,0,0), (255,0,0), (198,156,100)

    # screen size
    screen_width = 500
    screen_height = 500
    
    # create a game window
    # screen = pygame.display.set_mode((screen_width, screen_height))
    
    # title
    # pygame.display.set_caption('InfinityGo-%d'%num_games)
    
    # define the size of the board and grids
    board_size = 600
    grid_size = board_size // 12
    num_rows = 8
    
    # record the coordinates of each grid
    x_coord, y_coord = [],[]
    
    # draw the board
    # screen.fill(brown)
    for i in range(num_rows+1):
        y = i*grid_size + 50
        # pygame.draw.line(screen, black, (50,y),(450,y),2)
        y_coord.append(y)
        
    for j in range(num_rows+1):
        x = j*grid_size + 50
        # pygame.draw.line(screen, black, (x,50),(x,450),2)
        x_coord.append(x)
    
    # update
    # pygame.display.flip()
    # return screen,(x_coord, y_coord)
    return (x_coord,y_coord)

# def go_board(stones, coords, screen,num_games):
def go_board(stones, coords):
    
    # draw(num_games)
    
    white, black, red, brown = (255, 255, 255), (0,0,0), (255,0,0), (198,156,100)
    board_size = 600
    grid_size = board_size // 12
    
    x_coord,y_coord = coords
    
    # draw the stones for each player    
    for [point,player] in stones:
        y_grid, x_grid = point
        
        # if player == 'black':
        #     pygame.draw.circle(screen, black, (x_coord[x_grid-1], y_coord[9-y_grid]), grid_size//2)
        # else:
        #     pygame.draw.circle(screen, white, (x_coord[x_grid-1], y_coord[9-y_grid]), grid_size//2)
            
    # pygame.display.flip() # update the window
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True
