labels = "bear, elephant, shark, snake, reptile, amphibian, goldfish, tiger shark, hammerhead, electric ray, stingray, cock, hen, ostrich, European fire salamander, common newt, eft, spotted salamander, axolotl, bullfrog, tree frog, tailed frog, loggerhead, leatherback turtle, mud turtle, terrapin, box turtle, banded gecko, common iguana, American chameleon, whiptail, agama, frilled lizard, alligator lizard, Gila monster, green lizard, African chameleon, Komodo dragon, African crocodile, American alligator, triceratops, thunder snake, ringneck snake, hognose snake, green snake, king snake, garter snake, water snake, vine snake, night snake, boa constrictor, rock python, Indian cobra, green mamba, sea snake, horned viper, diamondback, sidewinder, jellyfish, sea anemone, brain coral, flatworm, nematode, conch, snail, slug, sea slug, chiton, chambered nautilus, Dungeness crab, rock crab, fiddler crab, king crab, American lobster, spiny lobster, crayfish, hermit crab, isopod, tusker, echidna, platypus, wallaby, koala, wombat, grey whale, killer whale, dugong, sea lion, timber wolf, white wolf, red wolf, coyote, dingo, dhole, African hunting dog, hyena, red fox, kit fox, Arctic fox, grey fox, cougar, lynx, leopard, snow leopard, jaguar, lion, tiger, cheetah, mongoose, meerkat, tiger beetle, ladybug, ground beetle, long-horned beetle, leaf beetle, dung beetle, rhinoceros beetle, weevil, fly, bee, ant, grasshopper, cricket, walking stick, cockroach, mantis, cicada, leafhopper, lacewing, dragonfly, damselfly, monarch, cabbage butterfly, sulphur butterfly, lycaenid, starfish, sea urchin, sea cucumber, wood rabbit, hare, Angora, hamster, porcupine, fox squirrel, marmot, beaver, guinea pig, sorrel, zebra, hog, wild boar, warthog, hippopotamus, ox, water buffalo, bison, ram, bighorn, ibex, hartebeest, impala, gazelle, Arabian camel, llama, weasel, mink, polecat, black-footed ferret, otter, skunk, badger, armadillo, three-toed sloth, orangutan, gorilla, chimpanzee, gibbon, siamang, guenon, patas, baboon, macaque, langur, colobus, proboscis monkey, marmoset, capuchin, howler monkey, titi, spider monkey, squirrel monkey, Madagascar cat, indri."
labels_list = labels.split(',')

cat_classes = [
    "Tabby", "Tiger cat", "Persian cat", "Siamese cat", "Egyptian cat",
    "Cougar", "Lynx", "Leopard", "Snow leopard", "Jaguar", "Lion", "Tiger", "Cheetah"
]
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
    'bear', 'polar bear', 'American black bear', 'ice bear', 'sloth bear',
    'elephant', 'African elephant', 'Indian elephant',
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

# List of subclasses to exclude (as previously defined for cat, dog, bird)
exclude_classes = cat_classes + dog_classes + bird_classes

# Filter out the excluded subclasses from the main animal list
remaining_animal_classes = [animal for animal in animal_classes if animal not in exclude_classes]

# Count the remaining animal classes
print("Number of animal classes excluding cats, dogs, and birds:", len(remaining_animal_classes))

# print("Number of cat subclasses:", len(cat_classes))
# print("Number of dog subclasses:", len(dog_classes))
# print("Number of bird subclasses:", len(bird_classes))
print(len(dog_classes) + len(cat_classes) + len(bird_classes))
