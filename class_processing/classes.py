CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

INTERSECTION_CLASSES = ['pizza', 'ski', 'plate', 'racket', 'vase', 'jean', 'screen', 'umbrella', 'pillow', 'wing', 
                            'banana', 'kite', 'plane', 'laptop', 'zebra', 'pole', 'desk', 'sock', 'pot', 'ear', 'orange', 'cup']

PERSON_CLASSES = ['guy', 'girl', 'child', 'boy', 'kid', 'lady', 'man', 'men', 'people', 'person', 'player', 'woman']

#PERSON_CLASSES = ['guy', 'girl', 'child', 'boy', 'kid', 'lady', 'man', 'men', 'people', 'person', 'player', 'woman',
    #                   'eye', 'ear', 'face', 'head', 'hair','mouth', 'neck', 'nose', 'arm']

# BREEDS_CLASSES = ['animal','bag','bird', 'bear', 'cow', 'giraffe', 'horse', 'sheep',
#                     'bus','car','cap','chair','elephant','wheel','window', 'dog', 'vehicle', 'phone']

second_classes = ['bird', 'cat', 'dog', 'bear', 'elephant', 'boat', 'bus', 'car', 'train', 
                    'truck', 'bottle', 'chair', 'clock', 'laptop', 'phone', 'tie', 'wheel',
                    'flower', 'fruit', 'vegetable', 'vehicle', 'animal']

need_more_classes = second_classes + PERSON_CLASSES

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
            'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
            'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
            'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
            'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
            'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

cat_classes = [
    "Tabby", "Tiger cat", "Persian cat", "Siamese cat", "Egyptian cat",
    "Cougar", "Lynx", "Leopard", "Snow leopard", "Jaguar", "Lion", "Tiger", "Cheetah"
]

bear_classes = ['bear', 'polar bear', 'American black bear', 'ice bear', 'sloth bear']
elephant_classes = ['elephant', 'African elephant', 'Indian elephant']

dog_classes = [
    "Chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese", "Shih-Tzu", "Blenheim spaniel", "Papillon", "Toy terrier",
    "Rhodesian ridgeback", "Afghan hound", "Basset", "Beagle", "Bloodhound", "Bluetick", "Black-and-tan coonhound", "Walker hound",
    "English foxhound", "Redbone", "Borzoi", "Irish wolfhound", "Italian greyhound", "Whippet", "Ibizan hound", "Norwegian elkhound",
    "Otterhound", "Saluki", "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier", "American Staffordshire terrier",
    "Bedlington terrier", "Border terrier", "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier",
    "Yorkshire terrier", "Wire-haired fox terrier", "Lakeland terrier", "Sealyham terrier", "Airedale", "Cairn", "Australian terrier",
    "Dandie Dinmont", "Boston bull", "Miniature schnauzer", "Giant schnauzer", "Standard schnauzer", "Scotch terrier", "Tibetan terrier",
    "Silky terrier", "Soft-coated wheaten terrier", "West Highland white terrier", "Lhasa", "Flat-coated retriever", "Curly-coated retriever",
    "Golden retriever", "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer", "Vizsla", "English setter",
    "Irish setter", "Gordon setter", "Brittany spaniel", "Clumber", "English springer", "Welsh springer spaniel", "Cocker spaniel",
    "Sussex spaniel", "Irish water spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois", "Briard", "Kelpie", "Komondor", "Old English sheepdog",
    "Shetland sheepdog", "Collie", "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd", "Doberman", "Miniature pinscher",
    "Greater Swiss Mountain dog", "Bernese mountain dog", "Appenzeller", "EntleBucher", "Boxer", "Bull mastiff", "Tibetan mastiff", "French bulldog",
    "Great Dane", "Saint Bernard", "Eskimo dog", "Malamute", "Siberian husky", "Dalmatian", "Affenpinscher", "Basenji", "Pug", "Leonberg", "Newfoundland",
    "Great Pyrenees", "Samoyed", "Pomeranian", "Chow", "Keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "Toy poodle", "Miniature poodle", "Standard poodle",
    "Mexican hairless"
]
bird_classes = [
    "Ostrich", "Robin", "Eagle", "Hummingbird", "Flamingo", "Pelican", "Albatross", "King penguin",
    "Black swan", "Drake", "Red-breasted merganser", "Goose"
]

animal_classes = [
    'shark', 'goldfish', 'tiger shark', 'hammerhead', 'electric ray', 'stingray',
    'snake', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake', 'night snake', 'boa constrictor', 'rock python', 'Indian cobra', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder',
    'reptile', 'banded gecko', 'common iguana', 'American chameleon', 'whiptail', 'agama', 'frilled lizard', 'alligator lizard', 'Gila monster', 'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile', 'American alligator',
    'amphibian', 'European fire salamander', 'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog', 'tree frog', 'tailed frog',
    'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus',
    'crab', 'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab',
    'lobster', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod',
    'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat',
    'whale', 'grey whale', 'killer whale',
    'dugong', 'sea lion',
    'canine', 'timber wolf', 'white wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena',
    'fox', 'red fox', 'kit fox', 'Arctic fox', 'grey fox',
    'feline', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah',
    'mongoose', 'meerkat',
    'insect', 'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly',
    'butterfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid',
    'starfish', 'sea urchin', 'sea cucumber',
    'rabbit', 'wood rabbit', 'hare', 'Angora',
    'rodent', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig',
    'ungulate', 'sorrel', 'zebra', 'hog', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'Arabian camel', 'llama',
    'weasel', 'mink', 'polecat', 'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth',
    'primate', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey', 'titi', 'spider monkey', 'squirrel monkey', 'Madagascar cat', 'indri'
]

items = [
    {
        "category": "Flowers",
        "items": ["Daisy", "Yellow lady's slipper"]
    },
    {
        "category": "Fruits",
        "items": ["Granny Smith", "Strawberry", "Orange", "Lemon", "Fig", "Pineapple", "Banana", "Jackfruit", "Custard apple", "Pomegranate"]
    },
    {
        "category": "Vegetables",
        "items": ["Zucchini", "Broccoli", "Cauliflower"]
    },
    {
        "category": "Vehicles",
        "items": ["Motorcycle", "Truck", "Bus"]
    }
]