# As an example, here is an implementation of
# the first problem "Ryerson Letter Grade":

"""
def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89:
        return 'A+'
    elif n > 84:
        return 'A'
    elif n > 79:
        return 'A-'
    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust
"""

"""
def is_ascending(items):

    if len(items) <= 1:
        return True

    for i in range(1, len(items)):
        if items[i] <= items[i-1]:
            return False

    return True


def riffle(items, out=True):

    itemsLen = len(items)
    halfLen = itemsLen // 2

    resultItems = [0] * len(items)

    for i in range(halfLen):

        if out:
            resultItems[2*i] = items[i]
            resultItems[2*i+1] = items[i + halfLen]
        else:
            resultItems[2*i+1] = items[i]
            resultItems[2*i] = items[i + halfLen]

    return resultItems

def only_odd_digits(n):

    while (n != 0):
        lastDigit = n % 10
        if lastDigit % 2 == 0:
            return False
        n = n // 10

    return True

def is_cyclops(n):

    if n == 0:
        return True

    # find size of n
    sizecount = 0
    ncpy = n
    while (ncpy != 0):
        ncpy = ncpy // 10
        sizecount += 1

    if sizecount % 2 == 0:
        return False

    eyeIndex = sizecount // 2

    # check right digits until the eye
    for i in range(eyeIndex):

        lastDigit = n % 10
        if lastDigit == 0:
            return False

        n = n // 10

    # check if digit is eye (0)
    lastDigit = n % 10
    if lastDigit != 0:
        return False

    n = n // 10

    # check left digits from eye to first digit
    for i in range(eyeIndex):
        lastDigit = n % 10
        if lastDigit == 0:
            return False

        n = n // 10

    return True

def domino_cycle(tiles):

    if len(tiles) == 0:
        return True

    pipValuesArr = []
    for dominoTile in tiles:
        pipValuesArr.append(dominoTile[0])
        pipValuesArr.append(dominoTile[1])

    for i in range(len(pipValuesArr)//2):
        if pipValuesArr[i*2] != pipValuesArr[i*2-1]:
            return False

    return True


def __colour_duo(colours):
    coloursList = ['y','r','b']

    # if same colours return that colour
    if colours[0] == colours[1]:
        return colours[0]

    # calculate colour with mix colours
    colourResult = coloursList.copy()
    if colours[0] in colourResult:
        colourResult.remove(colours[0])
    if colours[1] in colourResult:
        colourResult.remove(colours[1])

    return colourResult[0]


def colour_trio(colours):

    if len(colours) == 1:
        return colours
    elif len(colours) == 2:
        return __colour_duo(colours)

    wcolours = colours
    while len(wcolours) != 1:
        wcoloursLen = len(wcolours)
        wocolours = wcolours
        wcolours = ""
        for i in range(wcoloursLen-1):
            wcolours += __colour_duo(wocolours[i] + wocolours[i+1])

    return wcolours


def count_dominators(items):

    if len(items) == 0:
        return 0

    count = 1
    largest = items[len(items)-1]
    for i in range(len(items)-1,-1,-1):
        if largest < items[i]:
            count += 1
            largest = items[i]

    return count



def extract_increasing(digits):

    resultList = []

    firstDigit = digits[0]
    wnum = int(firstDigit)
    resultList.append(wnum)
    fi = 1

    for i in range(1, len(digits)):
        num = int(digits[fi:i+1])
        if wnum < num:
            wnum = num
            fi = i+1
            resultList.append(wnum)

    return resultList

def words_with_letters(words, letters):
    expectedResults = []
    for word in words:
        letter = letters[0]
        letterIndex = 0
        for i in range(len(word)):
            if word[i] == letter:
                letterIndex += 1
                if letterIndex == len(letters):
                    expectedResults.append(word)
                    break
                letter = letters[letterIndex]

    return expectedResults


directions_mapping = {'NORTH': 0, 'EAST': 1, 'SOUTH': 2, 'WEST': 3}
directions_inv_mapping = {v: k for k, v in directions_mapping.items()}

def __taxi_is_turn(move):
    if move in ['R','L']:
        return True
    else:
        return False

def __taxi_move_forward(direction, position):
    global directions_inv_mapping
    if directions_inv_mapping[direction] == 'NORTH':
        position[1] += 1
    elif directions_inv_mapping[direction] == 'SOUTH':
        position[1] -= 1
    elif directions_inv_mapping[direction] == 'EAST':
        position[0] += 1
    elif directions_inv_mapping[direction] == 'WEST':
        position[0] -= 1
    return position

def __taxi_get_new_direction(direction, turn):
    global directions_mapping
    if turn == 'L':
        direction -= 1
        if direction == -1:
            direction = directions_mapping['WEST']
    elif turn == 'R':
        direction += 1
        if direction == 4:
            direction = directions_mapping['NORTH']
    else:
        raise TypeError('Invalid Turn Input')
    return direction

def taxi_zum_zum(moves):
    global directions_mapping
    global directions_inv_mapping
    taxi_position = [0, 0]
    taxi_direction = directions_mapping['NORTH']
    for move in moves:
        if __taxi_is_turn(move):
            taxi_direction = __taxi_get_new_direction(taxi_direction, move)
        elif move == 'F':
            taxi_position = __taxi_move_forward(taxi_direction, taxi_position)

    return tuple(taxi_position)

def give_change(amount, coins):
    expectedResults = []
    for coin in coins:
        r = amount % coin
        s = (amount - r) // coin
        if s == 0:
            continue
        for i in range(s):
            expectedResults.append(coin)
        amount = r
        if amount == 0:
            break
    return expectedResults

def __safe_squares_get_mark_rook_lines(board, rook):
    for ri in range(len(board)):
        board[ri][rook[1]] = 1
    for ci in range(len(board[rook[0]])):
        board[rook[0]][ci] = 1
    return board

def __safe_squares_count(board):
    countNotSafeSquares = 0
    for row in board:
        for column in row:
            if column == 1:
                countNotSafeSquares += 1
    boardSize = len(board) ** 2
    safeSquares = boardSize - countNotSafeSquares
    return safeSquares

def safe_squares_rooks(n, rooks):
    board = []
    for i in range(n):
        board += [[0] * n]
    for rook in rooks:
        board = __safe_squares_get_mark_rook_lines(board, rook)
    return __safe_squares_count(board)
    
def words_with_given_shape(words, shape):
    expectedResults = []
    shapeSize = len(shape)
    for word in words:
        wordLen = len(word)
        if wordLen - 1 != shapeSize:
            continue
        wordMatchesShape = True
        for i in range(len(word)-1):
            fl = ord(word[i])
            sl = ord(word[i+1])
            shapeBehavior = 0
            if fl > sl:
                shapeBehavior = -1
            elif fl == sl:
                shapeBehavior = 0
            else:
                shapeBehavior = 1
            if shape[i] != shapeBehavior:
                wordMatchesShape = False
                break
        if wordMatchesShape:
            expectedResults.append(word)

    return expectedResults

def __is_left_handed_configuration(pips):
    pipsTuple = tuple(pips)
    leftHandConfigurations = ((1,2,3),(2,3,1),(3,1,2))
    for lhconfig in leftHandConfigurations:
        if lhconfig == pipsTuple:
            return True
    return False

def is_left_handed(pips):
    flipped = False
    for i in range(len(pips)):
        if pips[i] > 3:
            pips[i] = 7 - pips[i]
            flipped = not flipped
    if (__is_left_handed_configuration(pips) and not flipped) \
            or (not __is_left_handed_configuration(pips) and flipped):
        return True
    else:
        return False

def __get_cards_with_suit(cards, suit):
    cardsWithSuit = []
    for card in cards:
        if card[1] == suit:
            cardsWithSuit.append(card)
    return cardsWithSuit

def winning_card(cards, trump=None):
    if trump == None:
        trump = cards[0][1]
    cardsWithTrumpSuit = __get_cards_with_suit(cards, trump)
    if len(cardsWithTrumpSuit) == 0:
        trump = cards[0][1]
        cardsWithTrumpSuit = __get_cards_with_suit(cards, trump)
    rank_mapping = {'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'jack':11,'queen':12,'king':13,'ace':14}
    highestTrumpSuitCard = cardsWithTrumpSuit[0]
    highestRank = rank_mapping[highestTrumpSuitCard[0]]
    for card in cardsWithTrumpSuit:
        if rank_mapping[card[0]] > highestRank:
            highestRank = rank_mapping[card[0]]
            highestTrumpSuitCard = card
    return highestTrumpSuitCard

def knight_jump(knight, start, end):
    knight = list(knight)
    for i in range(len(start)):
        canJump = False
        for move in knight:
            if start[i] + move == end[i] or start[i] - move == end[i]:
                canJump = True
                knight.remove(move)
                break
        if not canJump:
            return False
    return True

def __seven_zero_get_sevens(digitmax=100000):
    numDigits = 1
    sevensNum = 0
    while numDigits <= digitmax:
        sevensNum = sevensNum * 10 + 7
        yield sevensNum
        numDigits += 1

def __seven_zero_get(digitmax=100000):
    numDigits = 1
    while numDigits <= digitmax:
        sevensStr = "7"
        zerosStr = "0"
        sevenZerosStr = ""
        for i in range(1, numDigits):
            sevenZerosStr = sevensStr*i + zerosStr*(numDigits-i)
            yield int(sevenZerosStr)
        numDigits += 1


def seven_zero(n):

    if n % 2 != 0 and n % 5 != 0:
        for i in __seven_zero_get_sevens():
            if i % n == 0:
                return i
    else:
        for i in __seven_zero_get():
            if i % n == 0:
                return i


def __can_balance_total_torque(items, left=True):
    totalTorque = 0
    if left:
        fulcrumxi = len(items) + 1
        for i in range(len(items)):
            deltax = fulcrumxi - (i+1)
            totalTorque += deltax*items[i]
    else:
        for i in range(len(items)):
            deltax = i + 1
            totalTorque += deltax * items[i]
    return totalTorque
def can_balance(items):

    if len(items) == 1:
        return 0

    for i in range(0, len(items)):
        leftItems = items[:i]
        rightItems = items[i+1:]
        ltorque = __can_balance_total_torque(leftItems, True)
        rtorque =  __can_balance_total_torque(rightItems, False)
        if ltorque == rtorque:
            return i
    return -1

def josephus(n, k):

    expectedResults = []
    zealotsList = [x + 1 for x in range(n)]
    killIndex = (k - 1) % len(zealotsList)

    while True:

        expectedResults.append(zealotsList[killIndex])
        zealotsList.pop(killIndex)

        if len(zealotsList) == 0:
            break

        killIndex -= 1
        killIndex += k
        killIndex = killIndex % len(zealotsList)

    return expectedResults

def group_and_skip(n, out, ins):
    expectedResults = []
    while n > 0:
        r = n % out
        expectedResults.append(r)
        groups = int((n - r) / out)
        n = groups * ins
    return expectedResults


def pyramid_blocks_sum_positive_integers(n):
    return round((n**2+n)//2)

def pyramid_blocks_sum_squares(b):
    return round((2*b**3+3*b**2+b)//6)

def pyramid_blocks(n, m, h):
    return h * n * m + pyramid_blocks_sum_positive_integers(h-1) * (n + m) + pyramid_blocks_sum_squares(h-1)


def __count_growlers_get_direction(animal):
    if animal[0] == 'd' or animal[0] == 'c':
        return 'left'
    elif animal[2] == 'd' or animal[2] == 'c':
        return 'right'

def __count_growlers_get_animal_counts(animals):
    animalCounts = {'dog': 0, 'cat': 0}
    for animal in animals:
        if animal == 'dog' or animal == 'god':
            animalCounts['dog'] += 1
        elif animal == 'cat' or animal == 'tac':
            animalCounts['cat'] += 1
    return animalCounts

def count_growlers(animals):
    count = 0
    for i in range(len(animals)):
        animal = animals[i]
        direction = __count_growlers_get_direction(animal)
        animalCounts = {'dog': 0, 'cat': 0}
        if direction == 'left':
            animalCounts = __count_growlers_get_animal_counts(animals[:i])
        elif direction == 'right':
            animalCounts = __count_growlers_get_animal_counts(animals[i+1:])
        if animalCounts['dog'] > animalCounts['cat']:
            count += 1
    return count


def __bulgarian_solitaire_is_steady_state(piles, k):
    for i in range(1, k+1):
        if i not in piles:
            return False
    return True
    
def bulgarian_solitaire(piles, k):
    count = 0
    while not __bulgarian_solitaire_is_steady_state(piles,k):
        pepplesForNewPile = len(piles)
        if 1 in piles:
            piles = list(filter((1).__ne__, piles))
        for p in range(len(piles)):
            piles[p] -= 1
        piles.append(pepplesForNewPile)
        count += 1
    return count


def scylla_or_charybdis(moves, n):
    stepCounts = {}
    for i in range(1, len(moves)):
        position = 0
        numsteps = 0
        for m in range(i-1, len(moves), i):
            move = moves[m]
            if move == '+':
                position += 1
            elif move == '-':
                position -= 1
            numsteps += 1
            if position == n or position == -n:
                stepCounts[i] = numsteps
                break
    smallestV = len(moves)
    smallestK = 0
    for k, v in stepCounts.items():
        if smallestV > v:
            smallestV = v
            smallestK = k
    return smallestK



def arithmetic_progression(items):

    if len(items) == 1:
        return (items[0],0,1)

    stride_map = {}
    for i in range(len(items)-1):
        for ii in range(i, len(items)):
            stride = items[ii]-items[i]
            if i not in stride_map:
                stride_map[i] = [stride]
            else:
                stride_map[i].append(stride)

    progression_map = {}
    for numi, strides in stride_map.items():
        num = items[numi]
        for stride in strides:

            n = 1
            cmpi = numi

            for i in range(numi, len(items)):
                if items[i] != items[cmpi] and items[i]-items[cmpi] == stride:
                    n += 1
                    cmpi = i
                elif items[i]-items[cmpi] > stride:
                    break

            if n not in progression_map:
                progression_map[n] = [{'startindex': numi, 'start': num, 'stride': stride}]
            else:
                progression_map[n].append({'startindex': numi, 'start': num, 'stride': stride})

    longest = 0
    for k, v in progression_map.items():
        if k > longest:
            longest = k

    longest_progression_list = progression_map[longest]
    progressionsWithLowestStart = [progression_map[longest][0]]
    lowestStart = progression_map[longest][0]['start']
    for progression in longest_progression_list:
        if lowestStart > progression['start']:
            lowestStart = progression['start']
            progressionsWithLowestStart = [progression]
        elif lowestStart == progression['start']:
            progressionsWithLowestStart.append(progression)

    progressionsWithLowestStride = [progressionsWithLowestStart[0]]
    lowestStride = progressionsWithLowestStart[0]['stride']
    for progression in progressionsWithLowestStart:
        if lowestStride > progression['stride']:
            lowestStride = progression['stride']
            progressionsWithLowestStride = [progression]

    return (progression_map[longest][0]['start'], progression_map[longest][0]['stride'], longest)

def tukeys_ninthers(items):
    if len(items) == 1:
        return items[0]
    return tukeys_ninthers([sorted(items[i-1:i+2])[1] for i in range(1, len(items), 3)])

def collect_numbers(perm):
    inv = perm.copy()
    for j in range(len(perm)):
        i = perm[j]
        inv[i] = j
    numloops = 1
    for i in range(1, len(inv)):
        if inv[i] < inv[i-1]:
            numloops += 1
    return numloops


def verify_betweenness(perm, constraints):
    inv = perm.copy()
    for j in range(len(perm)):
        inv[perm[j]] = j
    for constraint in constraints:
        ai, bi, ci = inv[constraint[0]], inv[constraint[1]], inv[constraint[2]]
        if not (ai < bi < ci or ai > bi > ci):
            return False
    return True
def count_troikas(items):
    from itertools import combinations
    dict = {}
    [ dict.setdefault(items[i], []).append(i) for i in range(len(items)) ]
    c = [dict[k] for k in dict if len(dict[k]) >= 3]
    count = 0
    clist = [item for sublist in (list(combinations(ci, 3)) for ci in c) for item in sublist]
    for k, j, i in clist:
        if k == j * 2 - i:
            count += 1
    return count

def __crag_score_is_crag(dice):
    sum = 0
    [sum := sum + x for x in dice]
    pairs = [(a, b) for idx, a in enumerate(dice) for b in dice[idx + 1:] if a == b]
    if sum == 13 and len(pairs) > 0:
        return True
    return False

def __crag_score_is_thirteen(dice):
    sum = 0
    [sum := sum + x for x in dice]
    if sum == 13:
        return True
    return False

def __crag_score_is_three_of_a_kind(dice):
    if dice.count(dice[0]) == 3:
        return True
    return False

def __crag_score_is_straight(dice):
    sdice = sorted(dice)
    if sdice in [[1, 2, 3], [4, 5, 6], [1, 3, 5], [2, 4, 6]]:
        return True
    return False

def __crag_score_is_ones_to_sixes(dice):
    sdice = sorted(dice)
    pairs = [(a, b) for idx, a in enumerate(sdice) for b in sdice[idx + 1:] if a == b]
    if len(pairs) == 0:
        return True, max(sdice)
    elif len(pairs) == 1 and pairs[0][0] + pairs[0][1] > sdice[2]:
        return True, pairs[0][0] + pairs[0][1]
    elif len(pairs) == 1 and sdice[2] != pairs[0][0]:
        return True, max(sdice)
    elif len(pairs) == 1 and sdice[2] == pairs[0][0]:
        return True, pairs[0][0] + pairs[0][1]
    else:
        return False, 0

def crag_score(dice):
    if __crag_score_is_crag(dice):
        return 50
    elif __crag_score_is_thirteen(dice):
        return 26
    elif __crag_score_is_three_of_a_kind(dice):
        return 25
    elif __crag_score_is_straight(dice):
        return 20
    if __crag_score_is_straight(dice):
        return 20
    isOnesToSixes, score = __crag_score_is_ones_to_sixes(dice)
    if isOnesToSixes:
        return score

def __two_summers(items, goal, i=0, j=None):
    j = len(items)-1 if j is None else j
    while i < j:
        x = items[i] + items[j]
        if x == goal:
            return True  # Okay, that's a solution.
        elif x < goal:
            i += 1  # Smallest element can't be part of solution.
        else:
            j -= 1  # Largest element can't be part of solution.
    return False

def three_summers(items, goal):
    for i in range(len(items)):
        if __two_summers(items[i+1:], goal-items[i]):
            return True
    return False


def __two_summer_squares(squares, goal, i=0, j=None):
    j = len(squares)-1 if j is None else j
    while i <= j:
        x = squares[i] + squares[j]
        if x == goal:
            return tuple(sorted([i+1, j+1], reverse=True))
        elif x < goal:
            i += 1
        else:
            j -= 1
    return None
    
def sum_of_two_squares(n):
    from math import sqrt, floor
    maxSquareRoot = floor(sqrt(n))
    squares_list = [ x**2 for x in range(1, maxSquareRoot+1) ]
    return __two_summer_squares(squares_list, n)

def __count_carries_get_num_list(n):
    list = []
    while n != 0:
        list.insert(0, n % 10)
        n = n // 10
    return list

def __count_carries_2(a, b):
    anumbers = __count_carries_get_num_list(a)
    bnumbers = __count_carries_get_num_list(b)
    topnumbers, botnumbers = anumbers, bnumbers
    if len(anumbers) > len(bnumbers):
        bnumbers = [0] * (len(anumbers) - len(bnumbers)) + bnumbers
        botnumbers = bnumbers
        topnumbers = anumbers
    elif len(bnumbers) > len(anumbers):
        anumbers = [0] * (len(bnumbers) - len(anumbers)) + anumbers
        botnumbers = anumbers
        topnumbers = bnumbers
    carries = 0
    for i in range(len(topnumbers)-1, -1, -1):
        if topnumbers[i] + botnumbers[i] >= 10:
            carries += 1
            topnumbers[i-1] += 1

    return carries

def count_carries(a, b):
    carries, carrieamount = 0, 0
    if b > a:
        a, b = b, a
    while a != 0:
        tnd = a % 10
        a = a // 10
        bnd = b % 10 + carrieamount
        b = b // 10
        if tnd + bnd >= 10:
            carries += 1
            carrieamount = 1
        else:
            carrieamount = 0

    return carries


def __count_carries_3(a, b):
    a = str(a)
    b = str(b)
    # Initialize the value of
    # carry to 0
    carry = 0

    # Counts the number of carry
    # operations
    count = 0

    # Initialize len_a and len_b
    # with the sizes of strings
    len_a = len(a)
    len_b = len(b)

    while (len_a != 0 or len_b != 0):

        # Assigning the ascii value
        # of the character
        x = 0
        y = 0
        if (len_a > 0):
            x = int(a[len_a - 1]) + int('0')
            len_a -= 1

        if (len_b > 0):
            y = int(b[len_b - 1]) + int('0')
            len_b -= 1

        # Add both numbers/digits
        sum = x + y + carry

        # If sum > 0, increment count
        # and set carry to 1
        if (sum >= 10):
            carry = 1
            count += 1

        # Else, set carry to 0
        else:
            carry = 0

    return count

def __count_carries_4(a, b):
    # f is the digitSum function
    f = lambda n:sum(map(int, str(n)))
    return int((f(a)+f(b)-f(a+b))/9)



from fractions import Fraction
def leibniz(heads, positions):
    previous_row = [heads[0]]
    current_row = [heads[1]]
    for i in range(1, len(heads)):
        for ii in range(1, i+1):
            current_row.append(previous_row[ii-1]-current_row[ii-1])
        if i + 1 < len(heads):
            previous_row = current_row
            current_row = [heads[i+1]]

    return [current_row[i] for i in positions]


def expand_intervals(intervals):
    if ',' not in intervals:
        if intervals == '':
            return []
        elif intervals.isdigit():
            return int(intervals)
    intranges = intervals.split(",")
    results = []
    for intrange in intranges:
        if '-' in intrange:
            rangeVals = intrange.split("-")
            lowerVal = int(rangeVals[0])
            higherVal = int(rangeVals[1]) + 1
            for i in range(lowerVal, higherVal):
                results.append(i)
        else:
            results.append(int(intrange))
    return results

def collapse_intervals(items):
    if len(items) == 0:
        return ''
    results = str(items[0])
    for i in range(1, len(items)):
        if items[i] - items[i-1] > 1:
            results += ',' + str(items[i])
        elif results.rfind(',') < results.rfind('-'):
            results = results[:results.rindex('-')] + '-' + str(items[i])
        else:
            results += '-' + str(items[i])
    return results

"""

"""

def prominences(height):

    if len(height) == 1:
        if height[0] == 0:
            return []
        else:
            return [(0,height[0],height[0])]

    results = []

    for i in range(len(height)-1):
        elevation = height[i]
        position = i
        prominence = 0
        for ii in range(i, len(height)-1):
            if height[ii] > elevation and prominence == 0:
                break
            elif height[ii] > elevation and prominence > 0:
                results.append((position, elevation, prominence))
                print(prominence)
            elif prominence > elevation - height[ii]:
                prominence = elevation - height[ii]

    results.append((position, elevation, elevation))

    return results

print(prominences([3, 1, 4]))

def prominences_2(height):

    if len(height) == 1:
        if height[0] == 0:
            return []
        else:
            return [(0,height[0],height[0])]

    results = []


    elevation = height[0]
    position = 0
    index = 0
    prominence = 0

    for h in height:
        if h > elevation and prominence != 0:
            results.append((position, elevation, prominence))
            prominence = 0
            position = index
            elevation = h
        elif h > elevation:
            elevation = h
            position = index
        elif h < elevation and prominence < elevation - h:
            prominence = elevation - h
        index += 1

    results.append((position, elevation, elevation))

    elevation = height[len(height)-1]
    position = len(height)-1
    index = len(height)-1
    prominence = 0
    for i in range(len(height)-1,-1,-1):
        h = height[i]
        if h > elevation and prominence != 0:
            results.append((position, elevation, prominence))
            prominence = 0
            position = index
            elevation = h
        elif h > elevation:
            elevation = h
            position = index
        elif h < elevation and prominence < elevation - h:
            prominence = elevation - h
        index -= 1

    return results

"""


def __candy_stable_state(candies):
    for candy in candies:
        if candy >= 2:
            return False
    return True

def candy_share(candies):
    count = 0
    while not __candy_stable_state(candies):
        newcandies = candies.copy()
        for i in range(len(candies)):
            if candies[i] >= 2:
                newcandies[i-1] += 1
                if i+1 >= len(candies):
                    newcandies[0] += 1
                else:
                    newcandies[i+1] += 1
                newcandies[i] -= 2
        count += 1
        candies = newcandies
    return count


def duplicate_digit_bonus(n):

    if n <= 9:
        return 0

    result = 0

    score = 0
    lastDigit = n % 10
    firstLastDigit = True
    while n >= 1:
        n = n // 10
        nextLastDigit = n % 10
        if lastDigit == nextLastDigit:
            if score == 0:
                score = 1
            else:
                score *= 10
        else:
            if firstLastDigit:
                score *= 2
            result += score
            score = 0
            firstLastDigit = False

        lastDigit = nextLastDigit

    result += score
    return result


def __nearest_smaller_get_right(num, items):
    for item in items:
        if item < num:
            return item

def nearest_smaller(items):

    if len(items) == 1:
        return items

    expected_results = items.copy()
    for i in range(len(items)):
        smaller = __nearest_smaller_get_right(items[i], items[i:])
        expected_results[i] = smaller
    return expected_results


"""

def ordinal_transform(seed, i):
    return None    

"""

def squares_intersect(s1, s2):
    if s1[0] + s1[2] >= s2[0] and \
        s1[0] <= s2[0] + s2[2] and \
        s1[1] + s1[2] >= s2[1] and \
        s1[1] <= s2[1] + s2[2]:
        return True
    return False


"""
def oware_move(board, house):
    return None

"""

def remove_after_kth(items, k=1):
    if k == 0:
        return []
    dict = {}
    newlist = []
    for item in items:
        if item not in dict:
            dict[item] = 1
            newlist.append(item)
        elif item in dict:
            if dict[item] < k:
                dict[item] += 1
                newlist.append(item)
    return newlist


"""
    def brussels_choice_step(n, mink, maxk):
"""






"""
 108
 - This one is hard to understand what to do
"""

"""
def cut_corners(points):

"""

"""
 109
 - This one sucks. It just gives error. Buggy problem
"""
"""
def fibonacci_word(k):
    kstr = str(k)
    if kstr[len(kstr)-1] == '0':
        return '0'
    else:
        return '1'
"""

"""
    Bonus Problems ---------------------------------------------------------------------
    110 - Hard. Requires backtraking and using recrusion to make efficient calls
"""

"""
import itertools
def reverse_110_generator(size):
    li10 = list(itertools.product([0, 1], repeat=size))
    for li in li10:
        yield list(li)

def reverse_110_too_slow(current):
    for result in reverse_110_generator(len(current)):
        retcurrent = [ 0 for x in result ]
        result.append(result[0])
        for i in range(0, len(result)-1):
            if result[i-1] == 1 and result[i] == 1 and result[i+1] == 1:
                retcurrent[i] = 0
            if result[i-1] == 1 and result[i] == 1 and result[i+1] == 0:
                retcurrent[i] = 1
            if result[i-1] == 1 and result[i] == 0 and result[i+1] == 1:
                retcurrent[i] = 1
            if result[i-1] == 1 and result[i] == 0 and result[i+1] == 0:
                retcurrent[i] = 0
            if result[i-1] == 0 and result[i] == 1 and result[i+1] == 1:
                retcurrent[i] = 1
            if result[i-1] == 0 and result[i] == 1 and result[i+1] == 0:
                retcurrent[i] = 1
            if result[i - 1] == 0 and result[i] == 0 and result[i + 1] == 1:
                retcurrent[i] = 1
            if result[i - 1] == 0 and result[i] == 0 and result[i + 1] == 0:
                retcurrent[i] = 0
        if retcurrent == current:
            return result[0:len(result)-1]
    return None

def reverse_110_backtrace(current, s):
    if current == 1:
        return s
    else:
        return [
            y + x
            for y in reverse_110_backtrace(1, s)
            for x in reverse_110_backtrace(current - 1, s)

        ]


def reverse_110(current):
    return reverse_110_backtrace(len(current), ["0","1"])
"""

#print(reverse_110_too_slow([1, 0, 1, 1, 0, 0, 0, 0]))
