import pygame

# ------------------------------------ #
# 环境渲染
# ------------------------------------ #

class MyRender:
    def __init__(self):
        self.px, self.py = 850, 700  # 行人坐标
        self.ax, self.ay = 50, 250  # 箭头坐标
        self.obs_list = []  # 保存所有障碍物的
        self.exit_list = []  # 出口坐标
        self.peo_list = {}  # 行人属性

        # ---------------------------------- #
        # 参数设置
        # ---------------------------------- #

        self.width = 950  # 窗口显示尺寸
        self.height = 950

        self.grid_size = 50  # 每个网格的size
        self.num_grid = 19  # 横纵方向的网格数

        root = 'D:/programmes/强化学习/图库/素材/'
        wall_fp = root + '棕墙.png'
        weilan_fp = root + '蓝色栅栏.png'
        person_fp = root + '行人.png'
        lay_fp = root + '躺.png'
        fire_fp = root + '燃脂.png'
        door_fp = root + '大门.png'
        plane_fp = root + '无人机.png'

        # 图片加载
        wall = pygame.image.load(wall_fp)  # 墙体图片加载
        door = pygame.image.load(door_fp)
        weilan = pygame.image.load(weilan_fp)
        person = pygame.image.load(person_fp)
        lay = pygame.image.load(lay_fp)
        fire = pygame.image.load(fire_fp)
        plane = pygame.image.load(plane_fp)

        # 尺寸调整
        self.wall = pygame.transform.scale(wall, (self.grid_size, self.grid_size))
        self.door = pygame.transform.scale(door, (self.grid_size, self.grid_size))
        self.weilan = pygame.transform.scale(weilan, (self.grid_size, self.grid_size))
        self.person = pygame.transform.scale(person, (self.grid_size, self.grid_size))
        self.lay = pygame.transform.scale(lay, (self.grid_size, self.grid_size))
        self.fire = pygame.transform.scale(fire, (self.grid_size, self.grid_size))
        self.plane = pygame.transform.scale(plane, (self.grid_size, self.grid_size))

    # 渲染
    def render(self):

        # ---------------------------------- #
        # 窗口化
        # ---------------------------------- #

        pygame.init()  # 初始化
        screen = pygame.display.set_mode((self.width, self.height))  # 设置窗口
        pygame.display.set_caption('Grid World')  # 窗口名称
        screen.fill((255, 255, 255))  # 窗口填充为白色

        # ------------------------------------- #
        # 构建墙体
        # ------------------------------------- #

        for i in range(self.num_grid//2-1):  # 第0行左
            pygame.display.set_icon(self.wall)  # 显示图片
            screen.blit(self.wall, (i*50, 0))
            self.obs_list.append((i*50, 0))
        for i in range(self.num_grid//2+2, self.num_grid):  # 第0行右
            pygame.display.set_icon(self.wall)  # 显示图片
            screen.blit(self.wall, (i*50, 0))
            self.obs_list.append((i*50, 0))

        for i in range(self.num_grid//2-1):  # 最后一行左
            pygame.display.set_icon(self.wall)  # 显示图片
            screen.blit(self.wall, (i*50, (self.num_grid-1)*50))
            self.obs_list.append((i*50, (self.num_grid-1)*50))
        for i in range(self.num_grid//2+2, self.num_grid):  # 最后一行右
            pygame.display.set_icon(self.wall)  # 显示图片
            screen.blit(self.wall, (i*50, (self.num_grid-1)*50))
            self.obs_list.append((i*50, (self.num_grid-1)*50))

        for j in range(1, self.num_grid-1):  # 最右侧一列
            pygame.display.set_icon(self.wall)  # 显示图片
            screen.blit(self.wall, ((self.num_grid-1)*50, j*50))
            self.obs_list.append(((self.num_grid-1)*50, j*50))

        for j in range(1, self.num_grid-1):  # 最左侧一列，偏上
            pygame.display.set_icon(self.wall)  # 显示图片
            screen.blit(self.wall, (0, j*50))
            self.obs_list.append((0, j*50))

        # ------------------------------------- #
        # 出口
        # ------------------------------------- #

        for i in range(self.num_grid//2-1, self.num_grid//2+2):  # 第0行左
            pygame.display.set_icon(self.door)  # 显示图片
            screen.blit(self.door, (i*50, 0))

        for i in range(self.num_grid//2-1, self.num_grid//2+2):  # 第0行左
            pygame.display.set_icon(self.door)  # 显示图片
            screen.blit(self.door, (i*50, 900))

        # ------------------------------------- #
        # 围栏
        # ------------------------------------- #

        # 横向
        for i in range(3, self.num_grid-3):  # 第1行
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (i*50, 150))

        for i in range(3, self.num_grid-3):  # 第2行
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (i*50, 350))

        for i in range(3, self.num_grid-3):  # 第3行
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (i*50, 550))

        for i in range(3, self.num_grid-3):  # 第4行
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (i*50, 750))

        # 纵向
        for j in range(4, 5):  # 列向-左上
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (150, j*50))
        for j in range(6, 7):  # 列向-左上
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (150, j*50))

        for j in range(12, 13):  # 列向-左下
            pygame.display.set_icon(self.weilan)
            screen.blit(self.weilan, (150, j*50))
        for j in range(14, 15):  # 列向-左下
            pygame.display.set_icon(self.weilan)
            screen.blit(self.weilan, (150, j*50))

        for j in range(4, 5):  # 列向-右上
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (750, j*50))
        for j in range(6, 7):  # 列向-右上
            pygame.display.set_icon(self.weilan)  # 显示图片
            screen.blit(self.weilan, (750, j*50))

        for j in range(12, 13):  # 列向-右下
            pygame.display.set_icon(self.weilan)
            screen.blit(self.weilan, (750, j*50))
        for j in range(14, 15):  # 列向-右下
            pygame.display.set_icon(self.weilan)
            screen.blit(self.weilan, (750, j*50))

        # 中间
        for j in range(4, 7):  # 列向-上
            pygame.display.set_icon(self.weilan)
            screen.blit(self.weilan, (450, j*50))

        for j in range(12, 15):  # 列向-下
            pygame.display.set_icon(self.weilan)
            screen.blit(self.weilan, (450, j*50))

        # ---------------------------------------- #
        # 火焰
        # ---------------------------------------- #
        pygame.display.set_icon(self.fire)  # 显示图片
        screen.blit(self.fire, (450, 450))

        # ---------------------------------------- #
        # 行人
        # ---------------------------------------- #

        pygame.display.set_icon(self.person)  # 显示图片
        screen.blit(self.person, (self.px, self.py))

        pygame.display.set_icon(self.lay)  # 显示图片
        screen.blit(self.lay, (300, 50))

        # ----------------------------------------- #
        # 无人机
        # ----------------------------------------- #
        pygame.display.set_icon(self.plane)  # 显示图片
        screen.blit(self.plane, (150, 450))


# ---------------------------------------- #
# 动画展示
# ---------------------------------------- #

env = MyRender()
env.render()
running = True
while running:
    # 获取事件
    for event in pygame.event.get():
        # QUIT常量代表退出，点击窗口的X退出
        if event.type == pygame.QUIT:
            running = False
    # 界面更新
    pygame.display.update()
