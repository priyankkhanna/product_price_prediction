###########################libraries import####################################3
from flask import Flask, request, redirect, url_for
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
from tensorflow.keras.utils import pad_sequences
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

#############################33loading neccesseties###############
my_scaler = joblib.load('scaler.gz')
model = load_model('fasttext_rmlse_0.28_mae_782_mape_0.27.h5')
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def generate_result(cat1,cat2,cat3,x_description,x_title):
    feed_data = {
        'x_title':[x_title],
        'x_description':[x_description],
        'category_1':[cat1],
        'category_2':[cat2],
        'category_3':[cat3]
    }
    dataset = pd.DataFrame(feed_data)
    dataset.x_title = dataset.x_title.str.replace('[^a-zA-Z0-9 ]', '')
    dataset.x_description = dataset.x_description.str.replace('[^a-zA-Z0-9 ]', '')
    max_title = 32
    max_description = 128
    dataset.x_title = tokenizer.texts_to_sequences(dataset.x_title)
    dataset.x_description = tokenizer.texts_to_sequences(dataset.x_description)

    #################converting input to feedable data################
    x_title = pad_sequences(dataset.x_title,maxlen = max_title)
    x_description = pad_sequences(dataset.x_description,maxlen = max_description)
    cat1 = dataset.category_1.to_numpy()
    cat2 = dataset.category_2.to_numpy()
    cat3 = dataset.category_3.to_numpy()
    feed_data = {
        'name':x_title,
        'item_desc':x_description,
        'category_1':cat1,
        'category_2':cat2,
        'category_3':cat3
    }
    #print(feed_data)
    #print(type(feed_data))
    #print(type(feed_data['name']))
    #####################generating results#######################
    y_pred = model.predict(feed_data)
    y_pred = my_scaler.inverse_transform(y_pred)
    y_pred = np.exp(y_pred)-1
    return y_pred[0][0]


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        title = request.form["title"]
        description = request.form["description"]
        cat1 = int(request.form["cat1"])
        cat2 = int(request.form["cat2"])
        cat3 = int(request.form["cat3"])
        output = round(generate_result(cat1,cat2,cat3,description,title),2)
        return f"""
        <html>
          <head>
            <style>
              form {{
                width: 500px;
                margin: 50px auto;
                text-align: center;
                padding: 20px;
                background-color: #f2f2f2;
                border-radius: 10px;
              }}
              input[type="text"] {{
                width: 100%;
                padding: 10px;
                margin-bottom: 20px;
                font-size: 18px;
                border-radius: 5px;
                border: 1px solid #ccc;
              }}
              input[type="submit"] {{
                padding: 10px 20px;
                font-size: 18px;
                border-radius: 5px;
                border: none;
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
              }}
              h3 {{
                text-align: center;
                margin-top: 50px;
              }}
            </style>
          </head>
          <body>
          <h4>Provided title - {title}</h4>
          <h4>Provided description - {description}</h4>
          <h4>Choosen category 1,2,3 are {cat1},{cat2},{cat3} respectively</h4>
            <h3>Expected output is {output} Rs</h3>
            <form method="post" action="/back">
              <input type="submit" value="Back">
            </form>
          </body>
        </html>
        """
    return """
        <html>
          <head>
            <style>
            select {{
                width: 300px;
                height: 40px;
                font-size: 18px;
                padding: 10px;
                margin: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                background-color: #fff;
                appearance: none;
                -webkit-appearance: none;
                -moz-appearance: none;
            }}

            select:hover {{
                border-color: #999;
                cursor: pointer;
            }}

            select:focus {{
                border-color: #000;
                outline: none;
            }}

            option {{
                font-size: 16px;
                padding: 10px;
                background-color: #fff;
            }}


              form {{
                width: 500px;
                margin: 50px auto;
                text-align: center;
                padding: 20px;
                background-color: #f2f2f2;
                border-radius: 10px;
              }}
              input[type="text"] {{
                width: 100%;
                padding: 10px;
                margin-bottom: 20px;
                font-size: 18px;
                border-radius: 5px;
                border: 1px solid #ccc;
              }}
              input[type="submit"] {{
                padding: 10px 20px;
                font-size: 18px;
                border-radius: 5px;
                border: none;
                background-color: #4CAF50;
                color: white;
                cursor: pointer;
              }}
              h3 {{
                text-align: center;
                margin-top: 50px;
              }}
            </style>
          </head>
          <body>
            <form method="post">

             <h3>Choose Category 1</h3>
            <select id="cat1" name="cat1" required>
                <option value="0">Bady and Kids</option><option value="1">Electronics</option><option value="2">Home and Furniture</option><option value="3">Men's wear</option><option value="4">Sports, Books and More</option><option value="5">Women's wear</option>
            </select>

            <h3>Choose Category 2</h3>
           <select id="cat2" name="cat2" required>
               <option value="0">Auto Accessories</option><option value="1">Baby Boy Clothing</option><option value="2">Baby Care</option><option value="3">Baby Girl Clothing</option><option value="4">Beauty and Grooming</option><option value="5">Bed Room Furniture</option><option value="6">Books </option><option value="7">Boys Clothing</option><option value="8">Camera Accessories</option><option value="9">Cameras</option><option value="10">Cleaning Supplies</option><option value="11">Computer Accessories</option><option value="12">Computer Peripherals</option><option value="13">DIY Furniture</option><option value="14">Desktop PCs</option><option value="15">Ethnic Bottoms</option><option value="16">Ethnic Wear</option><option value="17">Exercise Fitness </option><option value="18">Festive Decor and Gifts</option><option value="19">Food Essentials</option><option value="20">Furnishing</option><option value="21">Gaming</option><option value="22">Gaming and Accessories</option><option value="23">Girls Clothing</option><option value="24">Health &Nutrition </option><option value="25">Health Care Appliances</option><option value="26">Home Décor</option><option value="27">Home Improvement</option><option value="28">Home Lighting</option><option value="29">Industrial &Scientific tools </option><option value="30">Kids Footwear </option><option value="31">Kitchen ,Cookware and Serveware</option><option value="32">Kitchen Storage</option><option value="33">Laptops</option><option value="34">Lingerie and sleepware</option><option value="35">Living Room Furniture</option><option value="36">Medical Supplies</option><option value="37">Men's Grooming</option><option value="38">Mobile Accessories</option><option value="39">Network Components</option><option value="40">Office and Study Furniture</option><option value="41">Personal Care Appliances</option><option value="42">Pet Supplies</option><option value="43">Raincoats and windcheaters</option><option value="44">School Supplies</option><option value="45">Smart Home Automation</option><option value="46">Smart Wearable Tech</option><option value="47">Speakers</option><option value="48">Sports</option><option value="49">Stationery</option><option value="50">Swim  and Beachwear</option><option value="51">Tablets</option><option value="52">Tableware and Dinnerware</option><option value="53">Ties, socks, caps and more</option><option value="54">Toys</option><option value="55">Western and maternity Wear</option>
           </select>

           <h3>Choose Category 3</h3>
          <select id="cat3" name="cat3" required>
              <option value="0">Ab Exercisers </option><option value="1">Academics </option><option value="2">Action Figures </option><option value="3">Apple Ipads</option><option value="4">Ayurvedic Supplements </option><option value="5">Baby Bath ,Hair and Skin Care </option><option value="6">Baby Bathing Accessories </option><option value="7">Baby Bedding</option><option value="8">Baby Cleaners and Detergents </option><option value="9">Baby Feeding Bottle and Accessories </option><option value="10">Baby Feeding Utensils and Accessories</option><option value="11">Baby Food </option><option value="12">Baby Gear </option><option value="13">Baby Gift Sets and Combo </option><option value="14">Baby Grooming </option><option value="15">Baby Medical and Health Care </option><option value="16">Baby Oral Care </option><option value="17">Baby Proofing and Safety </option><option value="18">Badminton </option><option value="19">Barware </option><option value="20">Bath Towels </option><option value="21">Bath and Spa</option><option value="22">Bathroom and Kitchen Fittings </option><option value="23">Bean Bags </option><option value="24">Beard Care and Grooming </option><option value="25">Beds </option><option value="26">Bedsheets </option><option value="27">Blankets </option><option value="28">Bluetooth Speakers</option><option value="29">Board Games </option><option value="30">Bp Monitors</option><option value="31">Bulbs</option><option value="32">Calculators </option><option value="33">Camping  and Hiking </option><option value="34">Car Audio /Video </option><option value="35">Car Mobile Accessories </option><option value="36">Card Holders </option><option value="37">Cardio Equipment </option><option value="38">Casseroles </option><option value="39">Cats </option><option value="40">Ceiling Lamp </option><option value="41">Chocolates </option><option value="42">Cleaning Supplies</option><option value="43">Clocks </option><option value="44">Coffee Mugs</option><option value="45">Coffee Tables </option><option value="46">Collapsible Wardrobes </option><option value="47">Cricket </option><option value="48">Curtains </option><option value="49">Cushions and Pillows </option><option value="50">Cycling </option><option value="51">DSLR and Mirrorless</option><option value="52">DTH Set Top Box</option><option value="53">Deodorants </option><option value="54">Deodorants and Perfumes </option><option value="55">Desk Organizers </option><option value="56">Desktop PCs</option><option value="57">Dhoti </option><option value="58">Dhoti Pants </option><option value="59">Diapers </option><option value="60">Diaries </option><option value="61">Dining Tables and Chairs </option><option value="62">Dinner Set </option><option value="63">Dogs </option><option value="64">Dolls and Doll Houses </option><option value="65">Dumbbells </option><option value="66">Educational Toys </option><option value="67">Emergency Lights </option><option value="68">Epilators </option><option value="69">External Hard Disks</option><option value="70">Festive Decor and Gifts</option><option value="71">Fish and Aquatics </option><option value="72">Flasks </option><option value="73">Floor Coverings</option><option value="74">Football </option><option value="75">Gaming Accessories </option><option value="76">Gaming Consoles </option><option value="77">Gaming Laptops</option><option value="78">Gaming and Accessories</option><option value="79">Gas Stoves </option><option value="80">Gifting Combos </option><option value="81">Grooming Kits </option><option value="82">Gym Gloves </option><option value="83">Hair Care </option><option value="84">Hair Dryers </option><option value="85">Hair Straightners </option><option value="86">Headphones & Headsets</option><option value="87">Health Drinks </option><option value="88">Helicopter and Drones </option><option value="89">Helmets and Riding Gears </option><option value="90">Home Gyms </option><option value="91">Home Utilities and Organizers </option><option value="92">Hot Water Bag </option><option value="93">Industrial Measurement Devices </option><option value="94">Industrial Testing Devices </option><option value="95">Innerwear </option><option value="96">Jeggings and Tights </option><option value="97">Key Chains </option><option value="98">Kids Footwear </option><option value="99">Kids Room Furniture </option><option value="100">Kitchen Containers </option><option value="101">Kitchen and Table Linen </option><option value="102">Kitchen tools </option><option value="103">Lab and Scientific Products </option><option value="104">Laptop Bags</option><option value="105">Laptop Skins and Decals</option><option value="106">Lawn and Gardening </option><option value="107">Leggings and Churidars </option><option value="108">Lens</option><option value="109">Literature and Fiction </option><option value="110">Lunch Box </option><option value="111">Lunch Boxes </option><option value="112">Lungi </option><option value="113">Make Up </option><option value="114">Mattresses </option><option value="115">Memory Cards</option><option value="116">Mobile Cables</option><option value="117">Mobile Cases</option><option value="118">Mobile Chargers</option><option value="119">Mobile Holders</option><option value="120">Monitors</option><option value="121">Mouse</option><option value="122">Musical Toys </option><option value="123">Non Fiction </option><option value="124">Nursing and Breast Feeding </option><option value="125">Nuts and Dry Fruits </option><option value="126">Outdoor Toys </option><option value="127">Packaging and Shipping Products </option><option value="128">Paintings </option><option value="129">Pans </option><option value="130">Party Supplies </option><option value="131">Pendrives</option><option value="132">Pens </option><option value="133">Perfumes </option><option value="134">Power Banks</option><option value="135">Pregnancy and Fertility Kits </option><option value="136">Pressure Cookers </option><option value="137">Printers and Ink Cartridges</option><option value="138">Protein Supplements </option><option value="139">Puzzles </option><option value="140">Raincoats and windcheaters</option><option value="141">Remote Control Toys </option><option value="142">Routers</option><option value="143">S.T.E.M Toys </option><option value="144">Safety Products </option><option value="145">Saree Shapewear and Petticoats </option><option value="146">School Bags </option><option value="147">School Combo Sets </option><option value="148">Screenguards</option><option value="149">Self -Help </option><option value="150">Sexual Wellness </option><option value="151">Shakers and Sippers </option><option value="152">Shapewear </option><option value="153">Shavers </option><option value="154">Shaving and Aftershave </option><option value="155">Sherwanis </option><option value="156">Shoe Racks </option><option value="157">Showpieces and Figurines </option><option value="158">Skating </option><option value="159">Skin Care </option><option value="160">Smart Bands</option><option value="161">Smart Door Locks </option><option value="162">Smart Glasses (VR )</option><option value="163">Smart Glasses (VR)</option><option value="164">Smart Headphones</option><option value="165">Smart Security System </option><option value="166">Smart Watches</option><option value="167">Sofa </option><option value="168">Sofa Beds</option><option value="169">Soft Toys </option><option value="170">Soundbars</option><option value="171">Sports and Action</option><option value="172">Stickers </option><option value="173">Support </option><option value="174">Sweets Store </option><option value="175">Swim and Beachwear</option><option value="176">Swimming </option><option value="177">TV Units </option><option value="178">Table Lamp </option><option value="179">Tawas </option><option value="180">Ties, socks, caps and more</option><option value="181">Tools and Measuring Equipments </option><option value="182">Toy Guns </option><option value="183">Trimmers </option><option value="184">Tripods</option><option value="185">Vehicle Lubricants </option><option value="186">Vitamin Supplements </option><option value="187">Wall Lamp </option><option value="188">Wall Shelves </option><option value="189">Wardrobes </option><option value="190">Water Bottles </option><option value="191">Weighing Scale</option><option value="192">Wipes </option><option value="193">Yoga Mat </option><option value="194">Young Readers </option><option value="195">snacks and beverages</option>
          </select>

          <h3>Enter description:</h3>
          <input type="text" name="description" id = "description" required>
          <h3>Enter title:</h3>
          <input type="text" name="title" id = "title" required>
              <input type="submit" value="Submit" required>
            </form>
          </body>
        </html>
    """

@app.route("/back", methods=["POST"])
def back():
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run()
