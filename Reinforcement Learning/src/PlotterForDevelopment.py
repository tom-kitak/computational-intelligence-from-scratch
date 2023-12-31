import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

num_of_steps_per_episode_toy_matrix_no_learning = np.array([[560, 1640, 558, 654, 484, 444, 480, 624, 2756, 2031],
                                        [2300, 1648, 446, 422, 1260, 1278, 1744, 932, 1204, 670],
                                        [1226, 372, 248, 1204, 550, 846, 1906, 420, 1422, 1576],
                                        [1752, 200, 292, 780, 1348, 936, 1830, 1886, 764, 538],
                                        [934, 338, 316, 2482, 226, 268, 102, 1198, 566, 1101],
                                        [1102, 2128, 692, 1390, 3460, 2986, 1348, 504, 490, 2201],
                                        [852, 1050, 452, 480, 1062, 320, 672, 636, 1168, 884],
                                        [868, 546, 3482, 1120, 610, 1568, 616, 364, 1418, 2094],
                                        [454, 1206, 5072, 452, 1156, 414, 1940, 3086, 1494, 1013],
                                        [1434, 902, 832, 1272, 342, 2108, 2614, 1748, 1528, 914]])

num_of_steps_per_episode_easy_matrix_no_learning = np.array([[14038, 1076, 1620, 6248, 1324, 12314, 5708, 9218, 2598, 4672],
                                                 [15472, 5098, 6498, 1808, 11180, 9032, 1350, 6328, 5538, 4804],
                                                 [1256, 10272, 1194, 3650, 3752, 10366, 13544, 4802, 1284, 3820],
                                                 [18456, 7084, 3380, 10638, 2220, 5552, 3728, 3042, 7332, 8801],
                                                 [6308, 13210, 1752, 13664, 3004, 2884, 15350, 10540, 7920, 9090],
                                                 [1834, 8426, 10628, 10470, 2868, 2524, 2462, 24362, 29824, 7074],
                                                 [28798, 16340, 4070, 6876, 1642, 1468, 6082, 788, 7992, 2572],
                                                 [8892, 31458, 1444, 17742, 22606, 2620, 14010, 11836, 2274, 3992],
                                                 [5294, 5076, 3844, 2092, 1308, 9372, 11304, 22268, 6594, 7262],
                                                 [6822, 16354, 9638, 4326, 6196, 7890, 4360, 10512, 8464, 2332]])


num_of_steps_per_episode_toy_matrix_learning = np.array([[350, 572, 214, 518, 620, 1172, 526, 208, 138, 746, 170, 328, 104, 184, 240, 122, 660, 100, 52, 256, 58, 354, 102, 52, 28, 166, 60, 30, 26, 28, 26, 28, 26, 36, 28, 30, 30, 30, 26, 28, 28, 28, 24, 28, 28, 26, 34, 36, 26, 26],
                                                         [1044, 1436, 1024, 1102, 646, 1014, 798, 680, 554, 80, 212, 202, 524, 142, 204, 168, 318, 246, 124, 178, 210, 150, 78, 34, 114, 70, 60, 34, 32, 26, 28, 26, 30, 26, 26, 26, 30, 28, 28, 26, 26, 24, 26, 28, 28, 26, 24, 28, 30, 26],
                                                         [668, 2840, 710, 1478, 278, 1220, 142, 548, 192, 192, 514, 298, 172, 88, 310, 110, 176, 62, 398, 178, 40, 36, 46, 40, 40, 26, 28, 28, 28, 24, 26, 24, 24, 24, 24, 26, 26, 24, 28, 28, 24, 26, 36, 30, 36, 26, 28, 26, 24, 28],
                                                         [348, 482, 416, 870, 72, 2752, 750, 124, 812, 334, 786, 74, 476, 76, 300, 312, 124, 128, 50, 104, 342, 256, 96, 76, 70, 56, 50, 28, 34, 50, 32, 24, 28, 30, 26, 24, 26, 30, 42, 24, 24, 28, 26, 26, 26, 46, 26, 30, 34, 28],
                                                         [444, 260, 998, 246, 1044, 266, 408, 838, 384, 268, 338, 576, 472, 300, 366, 156, 188, 54, 44, 46, 98, 126, 66, 56, 200, 28, 26, 36, 30, 30, 26, 26, 30, 30, 30, 30, 28, 32, 26, 26, 30, 28, 34, 28, 28, 28, 32, 26, 28, 30],
                                                         [340, 1146, 284, 264, 2064, 806, 1656, 614, 744, 140, 150, 214, 622, 200, 70, 128, 320, 218, 144, 154, 34, 56, 58, 24, 24, 26, 30, 26, 24, 24, 24, 24, 24, 24, 24, 26, 28, 28, 26, 26, 28, 24, 30, 24, 24, 28, 26, 30, 26, 26],
                                                         [912, 820, 608, 292, 720, 540, 256, 112, 184, 72, 206, 216, 178, 186, 116, 144, 356, 76, 38, 326, 44, 46, 140, 56, 38, 24, 28, 26, 26, 24, 24, 42, 26, 24, 24, 24, 30, 24, 24, 32, 32, 24, 26, 26, 26, 26, 24, 28, 24, 28],
                                                         [2996, 1386, 1324, 454, 206, 246, 362, 1662, 1260, 140, 784, 342, 766, 118, 236, 162, 438, 140, 78, 220, 154, 140, 122, 42, 38, 124, 84, 30, 24, 32, 28, 26, 24, 28, 24, 24, 24, 30, 24, 26, 26, 26, 28, 28, 32, 24, 24, 36, 24, 24],
                                                         [3356, 306, 576, 402, 1350, 1030, 212, 398, 406, 446, 46, 468, 566, 40, 114, 326, 112, 156, 278, 64, 32, 80, 36, 32, 26, 26, 24, 24, 28, 28, 26, 28, 24, 24, 26, 34, 28, 26, 26, 26, 24, 24, 26, 26, 26, 24, 24, 24, 26, 26],
                                                         [414, 4350, 1000, 192, 244, 506, 498, 728, 116, 274, 258, 178, 132, 144, 168, 146, 114, 92, 130, 34, 38, 40, 84, 48, 28, 24, 28, 28, 26, 26, 32, 30, 24, 26, 28, 24, 30, 30, 24, 24, 24, 24, 26, 30, 26, 34, 30, 26, 28, 24]])

num_of_steps_per_episode_easy_matrix_learning = np.array([[12944, 7232, 3604, 8696, 1092, 854, 6440, 6860, 1494, 1292, 1720, 5584, 5092, 578, 2874, 1802, 1094, 796, 4390, 3094, 5888, 286, 2176, 3326, 4278, 314, 6978, 3232, 412, 214, 238, 960, 928, 400, 160, 554, 2116, 608, 108, 116, 114, 72, 96, 74, 46, 182, 42, 46, 40, 46],
                                                          [2150, 12412, 11264, 3474, 2688, 1260, 2194, 1662, 2580, 4620, 396, 5354, 1782, 3930, 550, 8484, 8822, 1490, 2466, 3012, 434, 406, 178, 4774, 2244, 3858, 442, 1872, 1102, 310, 828, 712, 682, 92, 640, 494, 208, 1890, 284, 1252, 334, 136, 230, 232, 88, 58, 48, 40, 44, 44],
                                                          [9934, 19872, 3282, 6306, 5728, 3644, 644, 960, 2338, 840, 1274, 6888, 5774, 418, 436, 974, 4162, 1776, 4916, 7666, 1510, 1538, 2142, 3844, 5844, 1576, 1042, 2426, 2234, 1272, 1284, 866, 98, 84, 1616, 1186, 568, 138, 206, 620, 2072, 68, 102, 2008, 68, 358, 48, 42, 38, 38],
                                                          [3208, 36974, 1822, 3040, 3230, 2484, 1062, 5962, 4646, 10872, 11898, 1056, 982, 1304, 5630, 2084, 498, 1922, 584, 6144, 758, 568, 2964, 474, 1420, 1016, 3200, 6042, 1488, 472, 414, 5456, 282, 114, 100, 158, 114, 508, 152, 74, 186, 88, 96, 104, 50, 38, 42, 44, 44, 44],
                                                          [7814, 15068, 4126, 10832, 2670, 1330, 16738, 2284, 2062, 2160, 4442, 2108, 10892, 3424, 930, 2432, 276, 956, 2390, 1266, 398, 930, 144, 574, 492, 402, 138, 2390, 380, 484, 3360, 482, 180, 76, 456, 296, 136, 104, 2886, 56, 72, 46, 58, 44, 46, 42, 40, 40, 44, 38],
                                                          [13642, 7676, 5536, 2794, 792, 14074, 3038, 530, 12274, 3364, 7710, 3934, 1052, 1716, 5880, 2290, 424, 1220, 1986, 506, 668, 1832, 628, 630, 152, 410, 214, 760, 2702, 1628, 84, 1810, 206, 372, 144, 2670, 92, 206, 334, 94, 62, 50, 52, 48, 98, 38, 42, 40, 38, 50],
                                                          [5272, 8390, 2496, 4218, 7366, 1728, 1662, 5740, 5904, 4574, 6060, 5026, 3628, 2436, 1098, 2396, 6022, 700, 2236, 1390, 672, 598, 1082, 366, 372, 430, 184, 304, 3042, 724, 516, 340, 154, 1046, 2308, 1986, 100, 1050, 50, 78, 56, 100, 46, 44, 62, 46, 166, 40, 38, 56],
                                                          [2428, 3936, 35380, 1922, 4104, 7452, 6256, 1436, 898, 5698, 6030, 8890, 504, 1404, 10018, 1938, 842, 956, 3084, 1400, 1424, 3818, 584, 774, 3208, 4684, 1810, 592, 486, 1574, 2050, 974, 1148, 346, 920, 464, 3288, 376, 1054, 528, 998, 9220, 206, 2864, 124, 138, 308, 68, 272, 246],
                                                          [2192, 9058, 5942, 1138, 5784, 17458, 10800, 2114, 1102, 9244, 7812, 1460, 7314, 3352, 6970, 2530, 2144, 3172, 1126, 132, 1296, 260, 234, 950, 884, 264, 4156, 194, 250, 1118, 778, 1512, 304, 218, 412, 146, 46, 120, 152, 86, 48, 50, 44, 46, 44, 42, 40, 42, 40, 42],
                                                          [12078, 3270, 3110, 1706, 1052, 1482, 978, 1280, 4986, 1544, 1296, 2020, 6016, 444, 4146, 2416, 354, 3878, 2524, 262, 1056, 840, 802, 784, 4698, 1256, 2182, 1184, 846, 1310, 798, 802, 68, 400, 982, 768, 382, 1858, 2588, 176, 612, 368, 84, 60, 578, 74, 64, 56, 54, 52]])

avg_num_of_steps_per_episode = np.mean(num_of_steps_per_episode_easy_matrix_learning, axis=0)

x = range(1, len(avg_num_of_steps_per_episode)+1)
plt.xticks(x)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(x, avg_num_of_steps_per_episode, color='r')
  
plt.xlabel("Episode Number")
plt.ylabel("Average Number Of Steps")
# plt.title("Average Number Of Steps In Each Episode For Toy Maze With Proper Learning")
plt.title("Average Number Of Steps In Each Episode For Easy Maze With Proper Learning")
        
plt.legend()
plt.show()