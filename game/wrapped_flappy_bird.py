import random
from itertools import cycle

import pygame

import game.flappy_bird_utils as flappy_bird_utils

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])

    



class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # player velocity, max velocity, downward accleration, accleration on flap
        self.pipeVelX = -3  # player's velocity along Y, default same as playerFlapped
        self.playerVelY = -0  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 8  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerFlapAcc = -8  # players speed on flapping  # FIXME придумайте, как оптимизировать шаг
        self.playerFlapped = False  # True when player flaps
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold

    def frame_step(self, input_actions):
        pygame.event.pump()
        
        # Инициализация награды
        reward = 0.0
        terminal = False
        score_updated = False  # Флаг для отслеживания обновления счета
        
        # Получаем параметры следующей трубы
        next_pipe = self.upperPipes[0]
        pipe_bottom = next_pipe['y'] + PIPE_HEIGHT
        pipe_center = pipe_bottom + PIPEGAPSIZE / 2
        player_center = self.playery + PLAYER_HEIGHT / 2
        distance_to_center = abs(player_center - pipe_center)
        
        # 1. Награда/штраф за положение
        max_distance = SCREENHEIGHT / 2
        normalized_distance = distance_to_center / max_distance
        center_reward = 2.0 * (1.0 - normalized_distance)
        distance_penalty = -1.5 * normalized_distance
        reward += center_reward + distance_penalty
        
        # 2. Штраф за края
        safe_margin = 50
        edge_penalty = 0.0
        
        if self.playery < safe_margin:
            edge_penalty = -3.0 * (1.0 - self.playery/safe_margin)
        elif self.playery > BASEY - PLAYER_HEIGHT - safe_margin:
            dist_to_bottom = BASEY - PLAYER_HEIGHT - self.playery
            edge_penalty = -3.0 * (1.0 - dist_to_bottom/safe_margin)
        
        reward += edge_penalty
        
        # 3. Выход за границы
        if self.playery <= 0 or self.playery >= BASEY - PLAYER_HEIGHT:
            reward = -10.0
            terminal = True
            self.__init__()
            return self._get_image_data(), reward, terminal, self.score

        # Прыжок
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True
                reward += 0.3

        # Проверка прохождения трубы 
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        pipeMidPos = next_pipe['x'] + PIPE_WIDTH / 2
        
        # Проверяем только первую трубу и только один раз
        if pipeMidPos - 2 <= playerMidPos < pipeMidPos + 2 and not score_updated:
            self.score += 1
            reward += 5.0
            score_updated = True  # Помечаем, что счет обновлен

        # Обновление игры
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # Физика
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
            
        if self.playerFlapped:
            self.playerFlapped = False
            self.playerRot = 45
            
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)

        # Движение труб
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # Добавление новых труб 
        if 0 < self.upperPipes[0]['x'] < 4:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # Удаление труб
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)
            score_updated = False  # Сбрасываем флаг при переходе к новой трубе

        # Проверка столкновений
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                            'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        
        if isCrash:
            reward = -8.0 if self.playery + PLAYER_HEIGHT >= BASEY - 1 else -5.0
            terminal = True
            self.__init__()
            return self._get_image_data(), reward, terminal, self.score

        # Отрисовка
        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        
        visibleRot = min(self.playerRot, self.playerRotThr)
        playerSurface = pygame.transform.rotate(IMAGES['player'][self.playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (self.playerx, self.playery))
        
        showScore(self.score)
        pygame.display.update()
        FPSCLOCK.tick(FPS)
    
        return self._get_image_data(), reward, terminal, self.score

    def _get_image_data(self):
        """Вспомогательный метод для получения данных изображения"""
        return pygame.surfarray.array3d(pygame.display.get_surface())
    
    def _getValidPipe(self):
        """Генерирует валидные трубы с проверкой на минимальный зазор"""
        while True:
            newPipe = getRandomPipe()
            # Проверяем, что зазор не слишком маленький
            if (newPipe[1]['y'] - (newPipe[0]['y'] + PIPE_HEIGHT)) >= PIPEGAPSIZE * 0.8:  # 80% от стандартного
                return newPipe


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs) - 1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
