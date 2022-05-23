#include "utils.hpp"		// custom function
#include "preprocess.hpp"	// preprocess plugin 
#include "logging.hpp"	
#include "calibrator.h"		// ptq

using namespace nvinfer1;
sample::Logger gLogger;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 1000;
static const int INPUT_C = 3;
static const int precision_mode = 32; // fp32 : 32, fp16 : 16, int8(ptq) : 8

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

// imagenet label name 1000
static std::vector<std::string> class_names{  // 1000 classes
	"tench Tinca tinca","goldfish Carassius auratus","great white shark white shark man-eater man-eating shark Carcharodon carcharias","tiger shark Galeocerdo cuvieri","hammerhead hammerhead shark","electric ray crampfish numbfish torpedo","stingray","cock","hen","ostrich Struthio camelus","brambling Fringilla montifringilla","goldfinch Carduelis carduelis","house finch linnet Carpodacus mexicanus","junco snowbird","indigo bunting indigo finch indigo bird Passerina cyanea","robin American robin Turdus migratorius","bulbul","jay","magpie","chickadee","water ouzel dipper","kite","bald eagle American eagle Haliaeetus leucocephalus","vulture","great grey owl great gray owl Strix nebulosa","European fire salamander Salamandra salamandra","common newt Triturus vulgaris","eft","spotted salamander Ambystoma maculatum","axolotl mud puppy Ambystoma mexicanum","bullfrog Rana catesbeiana","tree frog tree-frog","tailed frog bell toad ribbed toad tailed toad Ascaphus trui","loggerhead loggerhead turtle Caretta caretta","leatherback turtle leatherback leathery turtle Dermochelys coriacea","mud turtle","terrapin","box turtle box tortoise","banded gecko","common iguana iguana Iguana iguana","American chameleon anole Anolis carolinensis",
	"whiptail whiptail lizard","agama","frilled lizard Chlamydosaurus kingi","alligator lizard","Gila monster Heloderma suspectum","green lizard Lacerta viridis","African chameleon Chamaeleo chamaeleon","Komodo dragon Komodo lizard dragon lizard giant lizard Varanus komodoensis","African crocodile Nile crocodile Crocodylus niloticus","American alligator Alligator mississipiensis","triceratops","thunder snake worm snake Carphophis amoenus","ringneck snake ring-necked snake ring snake","hognose snake puff adder sand viper","green snake grass snake","king snake kingsnake","garter snake grass snake","water snake","vine snake","night snake Hypsiglena torquata","boa constrictor Constrictor constrictor","rock python rock snake Python sebae","Indian cobra Naja naja","green mamba","sea snake","horned viper cerastes sand viper horned asp Cerastes cornutus","diamondback diamondback rattlesnake Crotalus adamanteus","sidewinder horned rattlesnake Crotalus cerastes","trilobite","harvestman daddy longlegs Phalangium opilio","scorpion","black and gold garden spider Argiope aurantia","barn spider Araneus cavaticus","garden spider Aranea diademata","black widow Latrodectus mactans","tarantula","wolf spider hunting spider","tick","centipede",
	"black grouse","ptarmigan","ruffed grouse partridge Bonasa umbellus","prairie chicken prairie grouse prairie fowl","peacock","quail","partridge","African grey African gray Psittacus erithacus","macaw","sulphur-crested cockatoo Kakatoe galerita Cacatua galerita","lorikeet","coucal","bee eater","hornbill","hummingbird","jacamar","toucan","drake","red-breasted merganser Mergus serrator","goose","black swan Cygnus atratus","tusker","echidna spiny anteater anteater","platypus duckbill duckbilled platypus duck-billed platypus Ornithorhynchus anatinus","wallaby brush kangaroo","koala koala bear kangaroo bear native bear Phascolarctos cinereus","wombat","jellyfish","sea anemone anemone","brain coral","flatworm platyhelminth","nematode nematode worm roundworm","conch","snail","slug","sea slug nudibranch","chiton coat-of-mail shell sea cradle polyplacophore","chambered nautilus pearly nautilus nautilus","Dungeness crab Cancer magister","rock crab Cancer irroratus","fiddler crab","king crab Alaska crab Alaskan king crab Alaska king crab Paralithodes camtschatica","American lobster Northern lobster Maine lobster Homarus americanus","spiny lobster langouste rock lobster crawfish crayfish sea crawfish","crayfish crawfish crawdad crawdaddy",
	"hermit crab","isopod","white stork Ciconia ciconia","black stork Ciconia nigra","spoonbill","flamingo","little blue heron Egretta caerulea","American egret great white heron Egretta albus","bittern","crane","limpkin Aramus pictus","European gallinule Porphyrio porphyrio","American coot marsh hen mud hen water hen Fulica americana","bustard","ruddy turnstone Arenaria interpres","red-backed sandpiper dunlin Erolia alpina","redshank Tringa totanus","dowitcher","oystercatcher oyster catcher","pelican","king penguin Aptenodytes patagonica","albatross mollymawk","grey whale gray whale devilfish Eschrichtius gibbosus Eschrichtius robustus","killer whale killer orca grampus sea wolf Orcinus orca","dugong Dugong dugon","sea lion","Chihuahua","Japanese spaniel","Maltese dog Maltese terrier Maltese","Pekinese Pekingese Peke","Shih-Tzu","Blenheim spaniel","papillon","toy terrier","Rhodesian ridgeback","Afghan hound Afghan","basset basset hound","beagle","bloodhound sleuthhound","bluetick","black-and-tan coonhound",
	"Walker hound Walker foxhound","English foxhound","redbone","borzoi Russian wolfhound","Irish wolfhound","Italian greyhound","whippet","Ibizan hound Ibizan Podenco","Norwegian elkhound elkhound","otterhound otter hound","Saluki gazelle hound","Scottish deerhound deerhound","Weimaraner","Staffordshire bullterrier Staffordshire bull terrier","American Staffordshire terrier Staffordshire terrier American pit bull terrier pit bull terrier","Bedlington terrier","Border terrier","Kerry blue terrier","Irish terrier","Norfolk terrier","Norwich terrier","Yorkshire terrier","wire-haired fox terrier","Lakeland terrier","Sealyham terrier Sealyham","Airedale Airedale terrier","cairn cairn terrier","Australian terrier","Dandie Dinmont Dandie Dinmont terrier","Boston bull Boston terrier","miniature schnauzer","giant schnauzer","standard schnauzer","Scotch terrier Scottish terrier Scottie","Tibetan terrier chrysanthemum dog","silky terrier Sydney silky","soft-coated wheaten terrier","West Highland white terrier","Lhasa Lhasa apso","flat-coated retriever","curly-coated retriever","golden retriever","Labrador retriever","Chesapeake Bay retriever","German short-haired pointer","vizsla Hungarian pointer","English setter",
	"Irish setter red setter","Gordon setter","Brittany spaniel","clumber clumber spaniel","English springer English springer spaniel","Welsh springer spaniel","cocker spaniel English cocker spaniel cocker","Sussex spaniel","Irish water spaniel","kuvasz","schipperke","groenendael","malinois","briard","kelpie","komondor","Old English sheepdog bobtail","Shetland sheepdog Shetland sheep dog Shetland","collie","Border collie","Bouvier des Flandres Bouviers des Flandres","Rottweiler","German shepherd German shepherd dog German police dog alsatian","Doberman Doberman pinscher","miniature pinscher","Greater Swiss Mountain dog","Bernese mountain dog","Appenzeller","EntleBucher","boxer","bull mastiff","Tibetan mastiff","French bulldog","Great Dane","Saint Bernard St Bernard","Eskimo dog husky","malamute malemute Alaskan malamute","Siberian husky","dalmatian coach dog carriage dog","affenpinscher monkey pinscher monkey dog","basenji","pug pug-dog","Leonberg","Newfoundland Newfoundland dog","Great Pyrenees","Samoyed Samoyede","Pomeranian","chow chow chow","keeshond","Brabancon griffon","Pembroke Pembroke Welsh corgi","Cardigan Cardigan Welsh corgi","toy poodle","miniature poodle","standard poodle","Mexican hairless",
	"timber wolf grey wolf gray wolf Canis lupus","white wolf Arctic wolf Canis lupus tundrarum","red wolf maned wolf Canis rufus Canis niger","coyote prairie wolf brush wolf Canis latrans","dingo warrigal warragal Canis dingo","dhole Cuon alpinus","African hunting dog hyena dog Cape hunting dog Lycaon pictus","hyena hyaena","red fox Vulpes vulpes","kit fox Vulpes macrotis","Arctic fox white fox Alopex lagopus","grey fox gray fox Urocyon cinereoargenteus","tabby tabby cat","tiger cat","Persian cat","Siamese cat Siamese","Egyptian cat","cougar puma catamount mountain lion painter panther Felis concolor","lynx catamount","leopard Panthera pardus","snow leopard ounce Panthera uncia","jaguar panther Panthera onca Felis onca","lion king of beasts Panthera leo","tiger Panthera tigris","cheetah chetah Acinonyx jubatus","brown bear bruin Ursus arctos","American black bear black bear Ursus americanus Euarctos americanus","ice bear polar bear Ursus Maritimus Thalarctos maritimus","sloth bear Melursus ursinus Ursus ursinus","mongoose","meerkat mierkat","tiger beetle","ladybug ladybeetle lady beetle ladybird ladybird beetle","ground beetle carabid beetle","long-horned beetle longicorn longicorn beetle","leaf beetle chrysomelid",
	"dung beetle","rhinoceros beetle","weevil","fly","bee","ant emmet pismire","grasshopper hopper","cricket","walking stick walkingstick stick insect","cockroach roach","mantis mantid","cicada cicala","leafhopper","lacewing lacewing fly","dragonfly darning needle devils darning needle sewing needle snake feeder snake doctor mosquito hawk skeeter hawk","damselfly","admiral","ringlet ringlet butterfly","monarch monarch butterfly milkweed butterfly Danaus plexippus","cabbage butterfly","sulphur butterfly sulfur butterfly","lycaenid lycaenid butterfly","starfish sea star","sea urchin","sea cucumber holothurian","wood rabbit cottontail cottontail rabbit","hare","Angora Angora rabbit","hamster","porcupine hedgehog","fox squirrel eastern fox squirrel Sciurus niger","marmot","beaver","guinea pig Cavia cobaya","sorrel","zebra","hog pig grunter squealer Sus scrofa","wild boar boar Sus scrofa","warthog","hippopotamus hippo river horse Hippopotamus amphibius","ox","water buffalo water ox Asiatic buffalo Bubalus bubalis","bison","ram tup","bighorn bighorn sheep cimarron Rocky Mountain bighorn Rocky Mountain sheep Ovis canadensis","ibex Capra ibex","hartebeest","impala Aepyceros melampus","gazelle","Arabian camel dromedary Camelus dromedarius",
	"llama","weasel","mink","polecat fitch foulmart foumart Mustela putorius","black-footed ferret ferret Mustela nigripes","otter","skunk polecat wood pussy","badger","armadillo","three-toed sloth ai Bradypus tridactylus","orangutan orang orangutang Pongo pygmaeus","gorilla Gorilla gorilla","chimpanzee chimp Pan troglodytes","gibbon Hylobates lar","siamang Hylobates syndactylus Symphalangus syndactylus","guenon guenon monkey","patas hussar monkey Erythrocebus patas","baboon","macaque","langur","colobus colobus monkey","proboscis monkey Nasalis larvatus","marmoset","capuchin ringtail Cebus capucinus","howler monkey howler","titi titi monkey","spider monkey Ateles geoffroyi","squirrel monkey Saimiri sciureus","Madagascar cat ring-tailed lemur Lemur catta","indri indris Indri indri Indri brevicaudatus","Indian elephant Elephas maximus","African elephant Loxodonta africana","lesser panda red panda panda bear cat cat bear Ailurus fulgens","giant panda panda panda bear coon bear Ailuropoda melanoleuca","barracouta snoek","eel",
	"coho cohoe coho salmon blue jack silver salmon Oncorhynchus kisutch","rock beauty Holocanthus tricolor","anemone fish","sturgeon","gar garfish garpike billfish Lepisosteus osseus","lionfish","puffer pufferfish blowfish globefish","abacus","abaya","academic gown academic robe judges robe","accordion piano accordion squeeze box","acoustic guitar","aircraft carrier carrier flattop attack aircraft carrier","airliner","airship dirigible","altar","ambulance","amphibian amphibious vehicle","analog clock","apiary bee house","apron","ashcan trash can garbage can wastebin ash bin ash-bin ashbin dustbin trash barrel trash bin","assault rifle assault gun","backpack back pack knapsack packsack rucksack haversack","bakery bakeshop bakehouse","balance beam beam","balloon","ballpoint ballpoint pen ballpen Biro","Band Aid","banjo","bannister banister balustrade balusters handrail","barbell","barber chair","barbershop","barn","barometer","barrel cask","barrow garden cart lawn cart wheelbarrow","baseball","basketball","bassinet","bassoon","bathing cap swimming cap","bath towel","bathtub bathing tub bath tub","beach wagon station wagon wagon estate car beach waggon station waggon waggon","beacon lighthouse beacon light pharos","beaker",
	"bearskin busby shako","beer bottle","beer glass","bell cote bell cot","bib","bicycle-built-for-two tandem bicycle tandem","bikini two-piece","binder ring-binder","binoculars field glasses opera glasses","birdhouse","boathouse","bobsled bobsleigh bob","bolo tie bolo bola tie bola","bonnet poke bonnet","bookcase","bookshop bookstore bookstall","bottlecap","bow","bow tie bow-tie bowtie","brass memorial tablet plaque","brassiere bra bandeau","breakwater groin groyne mole bulwark seawall jetty","breastplate aegis egis","broom","bucket pail","buckle","bulletproof vest","bullet train bullet","butcher shop meat market","cab hack taxi taxicab","caldron cauldron","candle taper wax light","cannon","canoe","can opener tin opener","cardigan","car mirror","carousel carrousel merry-go-round roundabout whirligig","carpenters kit tool kit","carton","car wheel","cash machine cash dispenser automated teller machine automatic teller machine automated teller automatic teller ATM","cassette","cassette player","castle","catamaran","CD player","cello violoncello","cellular telephone cellular phone cellphone cell mobile phone","chain","chainlink fence","chain mail ring mail mail chain armor chain armour ring armor ring armour","chain saw chainsaw",
	"chest","chiffonier commode","chime bell gong","china cabinet china closet","Christmas stocking","church church building","cinema movie theater movie theatre movie house picture palace","cleaver meat cleaver chopper","cliff dwelling","cloak","clog geta patten sabot","cocktail shaker","coffee mug","coffeepot","coil spiral volute whorl helix","combination lock","computer keyboard keypad","confectionery confectionary candy store","container ship containership container vessel","convertible","corkscrew bottle screw","cornet horn trumpet trump","cowboy boot","cowboy hat ten-gallon hat","cradle","crane","crash helmet","crate","crib cot","Crock Pot","croquet ball","crutch","cuirass","dam dike dyke","desk","desktop computer","dial telephone dial phone","diaper nappy napkin","digital clock","digital watch","dining table board","dishrag dishcloth","dishwasher dish washer dishwashing machine","disk brake disc brake","dock dockage docking facility","dogsled dog sled dog sleigh","dome","doormat welcome mat","drilling platform offshore rig","drum membranophone tympan","drumstick","dumbbell","Dutch oven","electric fan blower","electric guitar","electric locomotive","entertainment center","envelope","espresso maker","face powder",
	"feather boa boa","file file cabinet filing cabinet","fireboat","fire engine fire truck","fire screen fireguard","flagpole flagstaff","flute transverse flute","folding chair","football helmet","forklift","fountain","fountain pen","four-poster","freight car","French horn horn","frying pan frypan skillet","fur coat","garbage truck dustcart","gasmask respirator gas helmet","gas pump gasoline pump petrol pump island dispenser","goblet","go-kart","golf ball","golfcart golf cart","gondola","gong tam-tam","gown","grand piano grand","greenhouse nursery glasshouse","grille radiator grille","grocery store grocery food market market","guillotine","hair slide","hair spray","half track","hammer","hamper","hand blower blow dryer blow drier hair dryer hair drier","hand-held computer hand-held microcomputer","handkerchief hankie hanky hankey","hard disc hard disk fixed disk","harmonica mouth organ harp mouth harp","harp","harvester reaper","hatchet","holster","home theater home theatre","honeycomb","hook claw","hoopskirt crinoline",
	"horizontal bar high bar","horse cart horse-cart","hourglass","iPod","iron smoothing iron","jack-o-lantern","jean blue jean denim","jeep landrover","jersey T-shirt tee shirt","jigsaw puzzle","jinrikisha ricksha rickshaw","joystick","kimono","knee pad","knot","lab coat laboratory coat","ladle","lampshade lamp shade","laptop laptop computer","lawn mower mower","lens cap lens cover","letter opener paper knife paperknife","library","lifeboat","lighter light igniter ignitor","limousine limo","liner ocean liner","lipstick lip rouge","Loafer","lotion","loudspeaker speaker speaker unit loudspeaker system speaker system","loupe jewelers loupe","lumbermill sawmill","magnetic compass","mailbag postbag","mailbox letter box","maillot","maillot tank suit","manhole cover","maraca","marimba xylophone","mask","matchstick","maypole","maze labyrinth","measuring cup","medicine chest medicine cabinet","megalith megalithic structure","microphone mike","microwave microwave oven","military uniform","milk can","minibus","miniskirt mini","minivan","missile","mitten","mixing bowl","mobile home manufactured home","Model T","modem","monastery","monitor","moped","mortar","mortarboard","mosque","mosquito net","motor scooter scooter",
	"mountain bike all-terrain bike off-roader","mountain tent","mouse computer mouse","mousetrap","moving van","muzzle","nail","neck brace","necklace","nipple","notebook notebook computer","obelisk","oboe hautboy hautbois","ocarina sweet potato","odometer hodometer mileometer milometer","oil filter","organ pipe organ","oscilloscope scope cathode-ray oscilloscope CRO","overskirt","oxcart","oxygen mask","packet","paddle boat paddle","paddlewheel paddle wheel","padlock","paintbrush","pajama pyjama pjs jammies","palace","panpipe pandean pipe syrinx","paper towel","parachute chute","parallel bars bars","park bench","parking meter","passenger car coach carriage","patio terrace","pay-phone pay-station","pedestal plinth footstall","pencil box pencil case","pencil sharpener","perfume essence","Petri dish","photocopier","pick plectrum plectron","pickelhaube","picket fence paling","pickup pickup truck","pier","piggy bank penny bank","pill bottle","pillow","ping-pong ball","pinwheel","pirate pirate ship","pitcher ewer","plane carpenters plane woodworking plane","planetarium","plastic bag","plate rack","plow plough","plunger plumbers helper","Polaroid camera Polaroid Land camera","pole","police van police wagon paddy wagon patrol wagon wagon black Maria",
	"poncho","pool table billiard table snooker table","pop bottle soda bottle","pot flowerpot","potters wheel","power drill","prayer rug prayer mat","printer","prison prison house","projectile missile","projector","puck hockey puck","punching bag punch bag punching ball punchball","purse","quill quill pen","quilt comforter comfort puff","racer race car racing car","racket racquet","radiator","radio wireless","radio telescope radio reflector","rain barrel","recreational vehicle RV R.V.","reel","reflex camera","refrigerator icebox","remote control remote","restaurant eating house eating place eatery","revolver six-gun six-shooter","rifle","rocking chair rocker","rotisserie","rubber eraser rubber pencil eraser","rugby ball","rule ruler","running shoe","safe","safety pin","saltshaker salt shaker","sandal","sarong","sax saxophone","scabbard","scale weighing machine","school bus","schooner","scoreboard","screen CRT screen","screw","screwdriver","seat belt seatbelt","sewing machine","shield buckler","shoe shop shoe-shop shoe store","shoji","shopping basket","shopping cart","shovel","shower cap","shower curtain","ski","ski mask","sleeping bag","slide rule slipstick","sliding door","slot one-armed bandit","snorkel","snowmobile","snowplow snowplough",
	"soap dispenser","soccer ball","sock","solar dish solar collector solar furnace","sombrero","soup bowl","space bar","space heater","space shuttle","spatula","speedboat","spider web spiders web","spindle","sports car sport car","spotlight spot","stage","steam locomotive","steel arch bridge","steel drum","stethoscope","stole","stone wall","stopwatch stop watch","stove","strainer","streetcar tram tramcar trolley trolley car","stretcher","studio couch day bed","stupa tope","submarine pigboat sub U-boat","suit suit of clothes","sundial","sunglass","sunglasses dark glasses shades","sunscreen sunblock sun blocker","suspension bridge","swab swob mop","sweatshirt","swimming trunks bathing trunks","swing","switch electric switch electrical switch","syringe","table lamp","tank army tank armored combat vehicle armoured combat vehicle","tape player","teapot","teddy teddy bear","television television system","tennis ball","thatch thatched roof","theater curtain theatre curtain","thimble","thresher thrasher threshing machine","throne","tile roof","toaster","tobacco shop tobacconist shop tobacconist","toilet seat","torch","totem pole","tow truck tow car wrecker","toyshop","tractor","trailer truck tractor trailer trucking rig rig articulated lorry semi",
	"tray","trench coat","tricycle trike velocipede","trimaran","tripod","triumphal arch","trolleybus trolley coach trackless trolley","trombone","tub vat","turnstile","typewriter keyboard","umbrella","unicycle monocycle","upright upright piano","vacuum vacuum cleaner","vase","vault","velvet","vending machine","vestment","viaduct","violin fiddle","volleyball","waffle iron","wall clock","wallet billfold notecase pocketbook","wardrobe closet press","warplane military plane","washbasin handbasin washbowl lavabo wash-hand basin","washer automatic washer washing machine","water bottle","water jug","water tower","whiskey jug","whistle","wig","window screen","window shade","Windsor tie","wine bottle","wing","wok","wooden spoon","wool woolen woollen","worm fence snake fence snake-rail fence Virginia fence","wreck","yawl","yurt","web site website internet site site","comic book","crossword puzzle crossword","street sign","traffic light traffic signal stoplight",	"book jacket dust cover dust jacket dust wrapper","menu","plate","guacamole","consomme","hot pot hotpot","trifle","ice cream icecream","ice lolly lolly lollipop popsicle","French loaf","bagel beigel","pretzel","cheeseburger","hotdog hot dog red hot","mashed potato","head cabbage","broccoli",
	"cauliflower","zucchini courgette","spaghetti squash","acorn squash","butternut squash","cucumber cuke","artichoke globe artichoke","bell pepper","cardoon","mushroom","Granny Smith","strawberry","orange","lemon","fig","pineapple ananas","banana","jackfruit jak jack","custard apple","pomegranate","hay","carbonara","chocolate sauce chocolate syrup","dough","meat loaf meatloaf","pizza pizza pie","potpie","burrito","red wine","espresso","cup","eggnog","alp","bubble","cliff drop drop-off","coral reef","geyser","lakeside lakeshore","promontory headland head foreland","sandbar sand bar","seashore coast seacoast sea-coast","valley vale","volcano","ballplayer baseball player","groom bridegroom","scuba diver","rapeseed","daisy","yellow ladys slipper yellow lady-slipper Cypripedium calceolus Cypripedium parviflorum","corn","acorn","hip rose hip rosehip","buckeye horse chestnut conker","coral fungus","agaric",
	"gyromitra","stinkhorn carrion fungus","earthstar","hen-of-the-woods hen of the woods Polyporus frondosus Grifola frondosa","bolete","ear spike capitulum","toilet tissue toilet paper bathroom tissue"
};

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
	float *gamma = (float*)weightMap[lname + ".weight"].values;
	float *beta = (float*)weightMap[lname + ".bias"].values;
	float *mean = (float*)weightMap[lname + ".running_mean"].values;
	float *var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };

	float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	return scale_1;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ stride, stride });
	conv1->setPaddingNd(DimsHW{ 1, 1 });

	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
	assert(conv2);
	conv2->setPaddingNd(DimsHW{ 1, 1 });

	IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

	IElementWiseLayer* ew1;
	if (inch != outch) {
		IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
		assert(conv3);
		conv3->setStrideNd(DimsHW{ stride, stride });
		IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
		ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
	}
	else {
		ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
	}
	IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
	assert(relu2);
	return relu2;
}

// Creat the engine using only the API and not any parser.
void createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
	std::cout << "==== model build start ====" << std::endl << std::endl;
	INetworkDefinition* network = builder->createNetworkV2(0U);

	std::map<std::string, Weights> weightMap = loadWeights("../Resnet18_py/resnet18.wts");
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ INPUT_H, INPUT_W, INPUT_C });
	assert(data);

	Preprocess preprocess{ maxBatchSize, INPUT_C, INPUT_H, INPUT_W, 0 };// Custom(preprocess) plugin 사용하기
	IPluginCreator* preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");// Custom(preprocess) plugin을 global registry에 등록 및 plugin Creator 객체 생성
	IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection*)&preprocess);// Custom(preprocess) plugin 생성
	IPluginV2Layer* preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);// network 객체에 custom(preprocess) plugin을 사용하여 custom(preprocess) 레이어 추가
	preprocess_layer->setName("preprocess_layer"); // layer 이름 설정
	ITensor* prep = preprocess_layer->getOutput(0);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*prep, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ 2, 2 });
	conv1->setPaddingNd(DimsHW{ 3, 3 });

	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);

	IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
	assert(pool1);
	pool1->setStrideNd(DimsHW{ 2, 2 });
	pool1->setPaddingNd(DimsHW{ 1, 1 });

	IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
	IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

	IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
	IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

	IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
	IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

	IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
	IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

	IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7, 7 });
	assert(pool2);
	pool2->setStrideNd(DimsHW{ 1, 1 });

	IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
	assert(fc1);

	fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*fc1->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1ULL << 29); // 512MB

	if (precision_mode == 16) {
		std::cout << "==== precision f16 ====" << std::endl << std::endl;
		config->setFlag(BuilderFlag::kFP16);
	}else if (precision_mode == 8) {
		std::cout << "==== precision int8 ====" << std::endl << std::endl;
		std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
		assert(builder->platformHasFastInt8());
		config->setFlag(BuilderFlag::kINT8);
		Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H,0, "../data_calib/", "../Int8_calib_table/resnet18_int8_calib.table", INPUT_BLOB_NAME);
		config->setInt8Calibrator(calibrator);
	}else {
		std::cout << "==== precision f32 ====" << std::endl << std::endl;
	}

	std::cout << "Building engine, please wait for a while..." << std::endl;
	IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
	std::cout << "==== model build done ====" << std::endl << std::endl;

	std::cout << "==== model selialize start ====" << std::endl << std::endl;
	std::ofstream p(engineFileName, std::ios::binary);
	if (!p) {
		std::cerr << "could not open plan output file" << std::endl << std::endl;
	}
	p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
	std::cout << "==== model selialize done ====" << std::endl << std::endl;
	engine->destroy();
	network->destroy();
	p.close();
	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}
}

int main()
{
	// 변수 선언 
	unsigned int maxBatchSize = 1;	// 생성할 TensorRT 엔진파일에서 사용할 배치 사이즈 값 
	bool serialize = false;			// Serialize 강제화 시키기(true 엔진 파일 생성)
	char engineFileName[] = "resnet18";

	char engine_file_path[256];
	sprintf(engine_file_path, "../Engine/%s_%d.engine", engineFileName, precision_mode);

	// 1) engine file 만들기 
	// 강제 만들기 true면 무조건 다시 만들기
	// 강제 만들기 false면, engine 파일 있으면 안만들고 
	//					   engine 파일 없으면 만듬
	bool exist_engine = false;
	if ((access(engine_file_path, 0) != -1)) {
		exist_engine = true;
	}

	if (!((serialize == false)/*Serialize 강제화 값*/ && (exist_engine == true) /*resnet18.engine 파일이 있는지 유무*/)) {
		std::cout << "===== Create Engine file =====" << std::endl << std::endl; // 새로운 엔진 생성
		IBuilder* builder = createInferBuilder(gLogger);
		IBuilderConfig* config = builder->createBuilderConfig();
		createEngine(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path); // *** Trt 모델 만들기 ***
		builder->destroy();
		config->destroy();
		std::cout << "===== Create Engine file =====" << std::endl << std::endl; // 새로운 엔진 생성 완료
	}

	// 2) engine file 로드 하기 
	char *trtModelStream{ nullptr };// 저장된 스트림을 저장할 변수
	size_t size{ 0 };
	std::cout << "===== Engine file load =====" << std::endl << std::endl;
	std::ifstream file(engine_file_path, std::ios::binary);
	if (file.good()) {
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[size];
		file.read(trtModelStream, size);
		file.close();
	}
	else {
		std::cout << "[ERROR] Engine file load error" << std::endl;
	}

	// 3) file에서 로드한 stream으로 tensorrt model 엔진 생성
	std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	IExecutionContext* context = engine->createExecutionContext();
	delete[] trtModelStream;

	void* buffers[2];
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	// GPU에서 입력과 출력으로 사용할 메모리 공간할당
	CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t)));
	CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float)));

	// 4) 입력으로 사용할 이미지 준비하기
	std::string img_dir = "../Resnet18_py/data/";
	std::vector<std::string> file_names;
	if (SearchFile(img_dir.c_str(), file_names) < 0) { // 이미지 파일 찾기
		std::cerr << "[ERROR] Data search error" << std::endl;
	}
	else {
		std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
	}
	cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
	cv::Mat ori_img;
	std::vector<uint8_t> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);	// 입력이 담길 컨테이너 변수 생성
	std::vector<float> outputs(OUTPUT_SIZE);
	for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t> 
		cv::Mat ori_img = cv::imread(file_names[idx]);
		cv::resize(ori_img, img, img.size()); // input size로 리사이즈
		memcpy(input.data(), img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
	}
	std::cout << "===== input load done =====" << std::endl << std::endl;

	uint64_t dur_time = 0;
	uint64_t iter_count = 1000;

	// CUDA 스트림 생성
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	//속도 측정에서 첫 1회 연산 제외하기 위한 계산
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
	context->enqueue(maxBatchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// 5) Inference 수행  
	for (int i = 0; i < iter_count; i++) {
		auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

		CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
		context->enqueue(maxBatchSize, buffers, stream, nullptr);
		CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		auto dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
		dur_time += dur;
		//std::cout << dur << " milliseconds" << std::endl;
	}
	dur_time /= 1000.f; //microseconds -> milliseconds
	// 6) 결과 출력
	std::cout << "==================================================" << std::endl;
	std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode <<std::endl;
	std::cout << iter_count << " th Iteration" << std::endl;
	std::cout << "Total duration time with data transfer : " << dur_time << " [milliseconds]" << std::endl;
	std::cout << "Avg duration time with data transfer : " << dur_time / (float)iter_count << " [milliseconds]" << std::endl;
	std::cout << "FPS : " << 1000.f / (dur_time / (float)iter_count) << " [frame/sec]" << std::endl;
	int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
	std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << std::endl;
	std::cout << "Class Name : " << class_names[max_index] << std::endl;
	std::cout << "==================================================" << std::endl;

	// Release stream and buffers ...
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}

// 2021-11-08 계산 결과값 출력
//==================================================
//Model : resnet18, Precision : 32
//100 th Iteration, Total dur time : 199 [milliseconds]
//Index : 388, Feature_value : 13.5538
//Class Name : giant panda panda panda bear coon bear Ailuropoda melanoleuca
//==================================================
//Model : resnet18, Precision : 16
//100 th Iteration, Total dur time : 58 [milliseconds]
//Index : 388, Feature_value : 13.5554
//Class Name : giant panda panda panda bear coon bear Ailuropoda melanoleuca
//==================================================
//Model : resnet18, Precision : 8 
//100 th Iteration, Total dur time : 40 [milliseconds]
//Index : 388, Feature_value : 13.9033
//Class Name : giant panda panda panda bear coon bear Ailuropoda melanoleuca
//==================================================
