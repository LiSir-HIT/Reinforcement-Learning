import pygame
import random
import math

# x向右为正，y向下为正
# ----------------------------------- #
# 窗口初始化
# ----------------------------------- #

game_over = False
score = 0  # 初始分数

width = 800
height = 600

pygame.init()  # 初始化
screen = pygame.display.set_mode((width, height))  # 设置窗口
pygame.display.set_caption('hit planes')  # 窗口名称

# 添加背景音效
bg_music = 'D:/zhouli/其他代码语言练习/pygame/picture/music.wav'
pygame.mixer_music.load(bg_music)
pygame.mixer.music.play(-1)  # 循环播放
# 添加爆炸音效
bao = 'D:/zhouli/其他代码语言练习/pygame/picture/bao.wav'
bao_sound = pygame.mixer.Sound(bao)

# ----------------------------------- #
# 创建对象
# ----------------------------------- #

# 背景
bg_fp = 'D:/zhouli/其他代码语言练习/pygame/picture/space.jpg'
bg = pygame.image.load(bg_fp)  # 图片加载
bg = pygame.transform.scale(bg, (width, height))  # 调整尺寸
pygame.display.set_icon(bg)  # 显示图片

# 玩家飞船
craft_fp = 'D:/zhouli/其他代码语言练习/pygame/picture/飞船.png'
icon = pygame.image.load(craft_fp)  # 图片加载
icon = pygame.transform.scale(icon, (width//10, height//10))  # 调整尺寸
pygame.display.set_icon(icon)  # 显示图片
# 玩家初始位置、速度
playerx = 360
playery = 500
playerStep = 0

# 创建敌人飞船
num_enemy = 6
enemy_fp = 'D:/zhouli/其他代码语言练习/pygame/picture/外星飞船.png'
class Enemy():
    def __init__(self):
        self.img = pygame.image.load(enemy_fp)  # 图片加载
        self.img = pygame.transform.scale(self.img, (width//10, height//10))  # 调整尺寸
        self.x = random.randint(200, 600)
        self.y = random.randint(50, 250)
        self.step = random.random() / 5  # 速度
    # 敌人死亡后位置恢复
    def reset(self):
        self.x = random.randint(200, 600)
        self.y = random.randint(50, 250)


# 创建敌人飞船
enemies = []  # 保存创建的飞船
for i in range(num_enemy):
    enemies.append(Enemy())

# ----------------------------------- #
# 运动
# ----------------------------------- #

# 敌人位置
def show_enemy():
    global game_over
    for e in enemies:  # 遍历所有飞船
        screen.blit(e.img, (e.x, e.y))
        e.x += e.step
        # 飞机碰到边界就改变方向
        if e.x > width - width//10 or e.x < 0:  # 右界限, 左界限
            e.step *= -1  # 反方向
            e.y += 50  # 下沉
            # 如果下沉到一定边就是敌人成功
            if e.y > 400:
                game_over = True
                print('game_over')
                enemies.clear()  # 清空敌人列表

# 玩家运动
def move_player():
    # 修改全局变量
    global playerx
    # 根据方向键调整飞船位置
    playerx += playerStep
    # 防止飞机出界
    if playerx > width - width//10:  # 右界限
        playerx = width - width//10
    if playerx < 0:  # 左界限
        playerx = 0

# ------------------------------------ #
# 计算子弹和敌人之间的距离
# ------------------------------------ #

def distance(bx,by, ex,ey):
    return math.sqrt((bx-ex)**2 + (by-ey)**2)

# ----------------------------------- #
# 子弹
# ----------------------------------- #

but_fp = 'D:/zhouli/其他代码语言练习/pygame/picture/子弹.png'
class Buttet():
    def __init__(self):
        self.img = pygame.image.load(but_fp)  # 图片加载
        self.img = pygame.transform.scale(self.img, (width//20, height//20))  # 调整尺寸
        # 子弹初始位置是玩家当前位置
        self.x = playerx + 20
        self.y = playery - 10
        self.step = 0.5  # 速度

    # 子弹击中敌人
    def hit(self):
        global score
        # 遍历所有敌人
        for e in enemies:
            # 计算敌人和子弹的距离
            if distance(self.x, self.y, e.x, e.y) < 20:
                # 击中--子弹消失
                bullets.remove(self)
                # 敌人位置复原
                e.reset()
                # 爆炸音效
                bao_sound.play()
                # 分数增加
                score += 1

# 保存现有的子弹
bullets = []

# 显示并移动子弹
def show_buttet():
    for b in bullets:  # 遍历所有子弹
        screen.blit(b.img, (b.x, b.y))  # 显示子弹
        b.hit()  # 攻击目标
        b.y -= b.step  # 向上移动
        # 飞出界就移除子弹
        if b.y < 0:
            bullets.remove(b)

# --------------------------------- #
# 显示分数
# --------------------------------- #

# font = pygame.font.SysFont('simsunnsimsun', 40)  # 字体
font = pygame.font.Font(None, 40)
def show_score():
    text = f'scores: {score}'
    # 渲染字体再显示
    score_render = font.render(text, True, (255,255,255))
    screen.blit(score_render, (10,10))

# --------------------------------- #
# 游戏结束提示
# --------------------------------- #
over_font = pygame.font.Font(None, 72)
def check_is_over():
    if game_over:
        text = 'Game Over'
        # 渲染字体再显示
        render = font.render(text, True, (255,0,0))
        screen.blit(render, (300,300))

# ----------------------------------- #
# 游戏循环
# ----------------------------------- #

# 循环
running = True
while running:

    # 绘制背景，锚点放在左上角(0,0)
    screen.blit(bg, (0,0))
    # 显示分数
    show_score()

    # 获取事件
    for event in pygame.event.get():
        # QUIT常量代表退出，点击窗口的X退出
        if event.type == pygame.QUIT:
            running = False

        # 按下键盘触发事件
        if event.type == pygame.KEYDOWN:
            # 判断哪一个按键
            if event.key == pygame.K_RIGHT:  # 向右的方向键
                playerStep = 0.5
            elif event.key == pygame.K_LEFT:  # 向左的方向键
                playerStep = -0.5
            elif event.key == pygame.K_SPACE:  # 按下空格
                print('发射子弹')
                # 创建子弹
                bullets.append(Buttet())

        # 抬起键盘触发事件
        if event.type == pygame.KEYUP:
            playerStep = 0

    # 绘制玩家飞船
    screen.blit(icon, (playerx, playery))
    move_player()
    # 绘制敌人飞船
    show_enemy()
    # 显示子弹
    show_buttet()

    # 每一帧都检查一下是否结束
    check_is_over()

    # 界面更新
    pygame.display.update()
