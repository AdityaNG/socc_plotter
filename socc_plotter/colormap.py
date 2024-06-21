import numpy as np


def socc_ade20k_label_colormap():
    """Creates a custom label colormap for visualization.

    Returns:
        A colormap for visualizing segmentation results.
    """
    return np.asarray(
        [
            [0, 0, 0],  # wall
            [0, 0, 0],  # building;edifice
            [0, 0, 0],  # sky
            [0, 0, 0],  # floor;flooring
            [0, 0, 0],  # tree
            [0, 0, 0],  # ceiling
            [125, 0, 125],  # road;route
            [0, 0, 0],  # bed
            [0, 0, 0],  # windowpane;window
            [0, 0, 0],  # grass
            [0, 0, 0],  # cabinet
            [0, 0, 0],  # sidewalk;pavement
            [0, 0, 255],  # person;individual;someone;somebody;mortal;soul
            [0, 0, 0],  # earth;ground
            [0, 0, 0],  # door;double;door
            [0, 0, 0],  # table
            [0, 0, 0],  # mountain;mount
            [0, 0, 0],  # plant;flora;plant;life
            [0, 0, 0],  # curtain;drape;drapery;mantle;pall
            [0, 0, 0],  # chair
            [255, 0, 0],  # car;auto;automobile;machine;motorcar
            [0, 0, 0],  # water
            [0, 0, 0],  # painting;picture
            [0, 0, 0],  # sofa;couch;lounge
            [0, 0, 0],  # shelf
            [0, 0, 0],  # house
            [0, 0, 0],  # sea
            [0, 0, 0],  # mirror
            [0, 0, 0],  # rug;carpet;carpeting
            [0, 0, 0],  # field
            [0, 0, 0],  # armchair
            [0, 0, 0],  # seat
            [0, 0, 0],  # fence;fencing
            [0, 0, 0],  # desk
            [0, 0, 0],  # rock;stone
            [0, 0, 0],  # wardrobe;closet;press
            [0, 0, 0],  # lamp
            [0, 0, 0],  # bathtub;bathing;tub;bath;tub
            [0, 0, 0],  # railing;rail
            [0, 0, 0],  # cushion
            [0, 0, 0],  # base;pedestal;stand
            [0, 0, 0],  # box
            [0, 0, 0],  # column;pillar
            [0, 0, 0],  # signboard;sign
            [0, 0, 0],  # chest;of;drawers;chest;bureau;dresser
            [0, 0, 0],  # counter
            [0, 0, 0],  # sand
            [0, 0, 0],  # sink
            [0, 0, 0],  # skyscraper
            [0, 0, 0],  # fireplace;hearth;open;fireplace
            [0, 0, 0],  # refrigerator;icebox
            [0, 0, 0],  # grandstand;covered;stand
            [0, 0, 0],  # path
            [0, 0, 0],  # stairs;steps
            [0, 0, 0],  # runway
            [0, 0, 0],  # case;display;case;showcase;vitrine
            [0, 0, 0],  # pool;table;billiard;table;snooker;table
            [0, 0, 0],  # pillow
            [0, 0, 0],  # screen;door;screen
            [0, 0, 0],  # stairway;staircase
            [0, 0, 0],  # river
            [0, 0, 0],  # bridge;span
            [0, 0, 0],  # bookcase
            [0, 0, 0],  # blind;screen
            [0, 0, 0],  # coffee;table;cocktail;table
            [0, 0, 0],  # toilet;can;commode;crapper;pot;potty;stool;throne
            [0, 0, 0],  # flower
            [0, 0, 0],  # book
            [0, 0, 0],  # hill
            [0, 0, 0],  # bench
            [0, 0, 0],  # countertop
            [0, 0, 0],  # stove;kitchen;stove;range;kitchen;range;cooking;stove
            [0, 0, 0],  # palm;palm;tree
            [0, 0, 0],  # kitchen;island
            [
                0,
                0,
                0,
            ],  # computer;computing;machine;computing
            [0, 0, 0],  # swivel;chair
            [255, 0, 0],  # boat
            [0, 0, 0],  # bar
            [0, 0, 0],  # arcade;machine
            [0, 0, 0],  # hovel;hut;hutch;shack;shanty
            [
                255,
                0,
                0,
            ],  # bus;autobus;coach;charabanc;double-decker
            [0, 0, 0],  # towel
            [0, 0, 0],  # light;light;source
            [255, 0, 0],  # truck;motortruck
            [0, 0, 0],  # tower
            [0, 0, 0],  # chandelier;pendant;pendent
            [0, 0, 0],  # awning;sunshade;sunblind
            [0, 0, 0],  # streetlight;street;lamp
            [0, 0, 0],  # booth;cubicle;stall;kiosk
            [
                0,
                0,
                0,
            ],  # television;television;receiver;television;
            [0, 0, 0],  # airplane;aeroplane;plane
            [0, 0, 0],  # dirt;track
            [0, 0, 0],  # apparel;wearing;apparel;dress;clothes
            [0, 0, 0],  # pole
            [0, 0, 0],  # land;ground;soil
            [0, 0, 0],  # bannister;banister;balustrade;balusters;handrail
            [0, 0, 0],  # escalator;moving;staircase;moving;stairway
            [0, 0, 0],  # ottoman;pouf;pouffe;puff;hassock
            [0, 0, 0],  # bottle
            [0, 0, 0],  # buffet;counter;sideboard
            [0, 0, 0],  # poster;posting;placard;notice;bill;card
            [0, 0, 0],  # stage
            [255, 0, 0],  # van
            [255, 0, 0],  # ship
            [0, 0, 0],  # fountain
            [
                0,
                0,
                0,
            ],  # conveyer;belt;conveyor;belt;conveyer;conveyor;transporter
            [0, 0, 0],  # canopy
            [0, 0, 0],  # washer;automatic;washer;washing;machine
            [0, 0, 0],  # plaything;toy
            [0, 0, 0],  # swimming;pool;swimming;bath;natatorium
            [0, 0, 0],  # stool
            [0, 0, 0],  # barrel;cask
            [0, 0, 0],  # basket;handbasket
            [0, 0, 0],  # waterfall;falls
            [0, 0, 0],  # tent;collapsible;shelter
            [0, 0, 0],  # bag
            [0, 0, 0],  # minibike;motorbike
            [0, 0, 0],  # cradle
            [0, 0, 0],  # oven
            [0, 0, 0],  # ball
            [0, 0, 0],  # food;solid;food
            [0, 0, 0],  # step;stair
            [0, 0, 0],  # tank;storage;tank
            [0, 0, 0],  # trade;name;brand;name;brand;marque
            [0, 0, 0],  # microwave;microwave;oven
            [0, 0, 0],  # pot;flowerpot
            [0, 0, 0],  # animal;animate;being;beast;brute;creature;fauna
            [255, 0, 255],  # bicycle;bike;wheel;cycle
            [0, 0, 0],  # lake
            [0, 0, 0],  # dishwasher;dish;washer;dishwashing;machine
            [0, 0, 0],  # screen;silver;screen;projection;screen
            [0, 0, 0],  # blanket;cover
            [0, 0, 0],  # sculpture
            [0, 0, 0],  # hood;exhaust;hood
            [0, 0, 0],  # sconce
            [0, 0, 0],  # vase
            (8, 122, 255),  # traffic;light;traffic;signal;stoplight
            [0, 0, 0],  # tray
            [
                0,
                0,
                0,
            ],  # ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;
            [0, 0, 0],  # fan
            [0, 0, 0],  # pier;wharf;wharfage;dock
            [0, 0, 0],  # crt;screen
            [0, 0, 0],  # plate
            [0, 0, 0],  # monitor;monitoring;device
            [0, 0, 0],  # bulletin;board;notice;board
            [0, 0, 0],  # shower
            [0, 0, 0],  # radiator
            [0, 0, 0],  # glass;drinking;glass
            [0, 0, 0],  # clock
            [0, 0, 0],  # flag
        ]
    )


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.

    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    return colormap


def create_ade20k_label_colormap():
    """Creates a label colormap used in ADE20K segmentation benchmark.

    Returns:
      A colormap for visualizing segmentation results.
    """
    return np.asarray(
        [
            [50, 50, 50],
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ],
        dtype=np.uint8,
    )
