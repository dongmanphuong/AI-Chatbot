import os
import json
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool

load_dotenv()

all_products = []
mock_cart = {}
mock_orders = {}
next_order_id = 1

user_product_interest_history = []

def fetch_products_from_api():
    print("Đang gọi API để lấy dữ liệu sản phẩm và thông tin...")
    time.sleep(1)

    # Đã bổ sung thêm trường 'brand' cho từng sản phẩm
    api_response_data = [
        {
            "id": "SP001",
            "name": "Samsung Galaxy S24 Ultra",
            "category": "Điện thoại",
            "brand": "Samsung", # <-- Đã thêm trường brand
            "description": "Điện thoại cao cấp của Samsung với màn hình Dynamic AMOLED 2X 6.8 inch, camera 200MP, pin 5000mAh, và bút S Pen tích hợp. Phiên bản mới nhất với AI cải tiến.",
            "price": 30000000,
            "features": ["Màn hình lớn", "Pin trâu", "Camera chất lượng cao", "Có S Pen", "Tính năng AI"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/normal-s24-ultra-titan-den.webp"
        },
        {
            "id": "SP002",
            "name": "iPhone 15 Pro Max",
            "category": "Điện thoại",
            "brand": "Apple", # <-- Đã thêm trường brand
            "description": "Điện thoại flagship của Apple với màn hình Super Retina XDR 6.7 inch, chip A17 Pro, hệ thống camera chuyên nghiệp với ống kính tele 5x. Hiệu năng vượt trội.",
            "price": 34000000,
            "features": ["Màn hình lớn", "Hiệu năng mạnh", "Camera chuyên nghiệp", "Hệ sinh thái Apple"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/iphone-15-pro-max-titan-den.webp"
        },
        {
            "id": "SP003",
            "name": "Dell XPS 15",
            "category": "Laptop",
            "brand": "Dell", # <-- Đã thêm trường brand
            "description": "Laptop cao cấp dành cho công việc với màn hình InfinityEdge 15.6 inch, bộ vi xử lý Intel Core i7 thế hệ mới, RAM 16GB, SSD 512GB. Thiết kế mỏng nhẹ, vỏ nhôm nguyên khối, thích hợp cho người sáng tạo nội dung.",
            "price": 28000000,
            "features": ["Màn hình đẹp", "Hiệu năng cao", "Thiết kế mỏng nhẹ", "Vỏ kim loại"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/dell-xps-15-9530.webp"
        },
        {
            "id": "SP004",
            "name": "Sony WH-1000XM5",
            "category": "Tai nghe",
            "brand": "Sony", # <-- Đã thêm trường brand
            "description": "Tai nghe chống ồn chủ động hàng đầu, âm thanh Hi-Res Audio, thời lượng pin 30 giờ, đệm tai mềm mại, kết nối đa điểm. Phù hợp cho việc du lịch và làm việc tập trung.",
            "price": 7000000,
            "features": ["Chống ồn tốt", "Âm thanh chất lượng cao", "Pin lâu", "Thoải mái khi đeo"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/tai-nghe-sony-wh-1000xm5.webp"
        },
        {
            "id": "SP005",
            "name": "Xiaomi Redmi Note 13 Pro",
            "category": "Điện thoại",
            "brand": "Xiaomi", # <-- Đã thêm trường brand
            "description": "Điện thoại tầm trung với màn hình AMOLED 120Hz, camera 200MP, pin 5000mAh, sạc nhanh 67W. Giá cả phải chăng, hiệu năng tốt trong phân khúc.",
            "price": 7500000,
            "features": ["Màn hình đẹp", "Camera cao", "Pin trâu", "Sạc nhanh"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/xiaomi-redmi-note-13-pro.webp"
        },
        {
            "id": "SP006",
            "name": "MacBook Air M3 2024",
            "category": "Laptop",
            "brand": "Apple", # <-- Đã thêm trường brand
            "description": "Laptop siêu mỏng nhẹ của Apple với chip M3 mạnh mẽ, thời lượng pin cả ngày, màn hình Liquid Retina sắc nét. Lý tưởng cho công việc di động và giải trí.",
            "price": 27000000,
            "features": ["Siêu mỏng nhẹ", "Pin cực lâu", "Hiệu năng mạnh", "Màn hình đẹp", "Hệ sinh thái Apple"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/macbook-air-m3-2024.webp"
        },
        {
            "id": "SP007",
            "name": "Apple Watch Series 9",
            "category": "Đồng hồ thông minh",
            "brand": "Apple", # <-- Đã thêm trường brand
            "description": "Đồng hồ thông minh cao cấp của Apple với chip S9 SiP, màn hình sáng hơn, tính năng sức khỏe và thể chất tiên tiến. Hỗ trợ cử chỉ chạm hai lần mới.",
            "price": 10000000,
            "features": ["Theo dõi sức khỏe", "Chống nước", "Màn hình sáng", "Kết nối di động"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/apple-watch-s9.webp"
        },
        {
            "id": "SP008",
            "name": "Loa Bluetooth JBL Flip 6",
            "category": "Loa",
            "brand": "JBL", # <-- Đã thêm trường brand
            "description": "Loa di động chống nước và bụi, âm thanh mạnh mẽ, thời lượng pin 12 giờ. Phù hợp cho các buổi tiệc ngoài trời.",
            "price": 2500000,
            "features": ["Chống nước", "Âm thanh lớn", "Pin lâu", "Di động"],
            "image_url": "https://cdn2.cellphones.com.vn/insecure/crop/350x350/filters:format(webp)/product/500x0/jbl-flip-6.webp"
        }
    ]

    policy_info_data = [
        {
            "id": "POLICY001",
            "type": "Chính sách",
            "subtype": "Đổi trả",
            "title": "Chính sách đổi trả",
            "content": "Bạn có thể đổi trả sản phẩm trong vòng 7 ngày kể từ ngày nhận hàng nếu sản phẩm bị lỗi kỹ thuật, không đúng mô tả, hoặc không đúng với đơn đặt hàng. Sản phẩm phải còn nguyên tem mác, chưa qua sử dụng và đầy đủ phụ kiện. Vui lòng liên hệ bộ phận hỗ trợ để được hướng dẫn chi tiết.",
            "image_url": "https://via.placeholder.com/150/FF5733/FFFFFF?text=DoiTra"
        },
        {
            "id": "POLICY002",
            "type": "Chính sách",
            "subtype": "Bảo hành",
            "title": "Chính sách bảo hành",
            "content": "Tất cả sản phẩm được bảo hành chính hãng theo thời gian quy định của nhà sản xuất, thường là 12 tháng. Vui lòng giữ hóa đơn mua hàng để được hỗ trợ bảo hành tốt nhất. Các trường hợp không bảo hành bao gồm hỏng hóc do va đập, vào nước, cháy nổ, hoặc can thiệp không đúng cách.",
            "image_url": "https://via.placeholder.com/150/33FF57/FFFFFF?text=BaoHanh"
        },
        {
            "id": "INFO001",
            "type": "Thông tin",
            "subtype": "Địa chỉ cửa hàng",
            "title": "Địa chỉ cửa hàng chính",
            "content": "Cửa hàng chính của chúng tôi tọa lạc tại: Số 123, Đường ABC, Quận XYZ, Thành phố Hồ Chí Minh. Giờ mở cửa: Thứ Hai - Chủ Nhật, từ 9:00 sáng đến 9:00 tối.",
            "image_url": "https://via.placeholder.com/150/3357FF/FFFFFF?text=DiaChi"
        },
        {
            "id": "INFO002",
            "type": "Thông tin",
            "subtype": "Liên hệ hỗ trợ",
            "title": "Liên hệ hỗ trợ",
            "content": "Để được hỗ trợ nhanh nhất, vui lòng gọi hotline của chúng tôi: 1900 1234 (hoạt động 24/7) hoặc gửi email về support@yourshop.com. Bạn cũng có thể truy cập trang Câu hỏi thường gặp trên website của chúng tôi.",
            "image_url": "https://via.placeholder.com/150/FFFF33/000000?text=HoTro"
        },
        {
            "id": "INFO003",
            "type": "Thông tin",
            "subtype": "Thời gian giao hàng",
            "title": "Thời gian giao hàng",
            "content": "Thời gian giao hàng tiêu chuẩn trong nội thành TP.HCM và Hà Nội là 1-2 ngày làm việc. Đối với các tỉnh thành khác, thời gian giao hàng có thể từ 3-5 ngày làm việc tùy thuộc vào địa điểm. Chúng tôi luôn cố gắng giao hàng nhanh nhất có thể sau khi đơn hàng được xác nhận.",
            "image_url": "https://via.placeholder.com/150/FF33FF/FFFFFF?text=GiaoHang"
        }
    ]

    print("Dữ liệu sản phẩm và thông tin đã được lấy từ API.")
    return api_response_data + policy_info_data


def load_products_data_from_api():
    global all_products

    products_and_info = fetch_products_from_api()

    documents = []
    all_products = [item for item in products_and_info if item.get("type") != "Chính sách" and item.get("type") != "Thông tin"]

    for item in products_and_info:
        if item.get("type") == "Chính sách" or item.get("type") == "Thông tin":
            content = (
                f"Tiêu đề: {item['title']}\n"
                f"Loại: {item['type']} - {item['subtype']}\n"
                f"Nội dung: {item['content']}\n"
                f"Ảnh minh họa: {item.get('image_url', 'Không có ảnh')}"
            )
            metadata = {
                "id": item['id'],
                "title": item['title'],
                "type": item['type'],
                "subtype": item['subtype'],
                "image_url": item.get('image_url', '')
            }
        else: # Đây là dữ liệu sản phẩm
            content = (
                f"ID: {item['id']}\n"
                f"Tên: {item['name']}\n"
                f"Danh mục: {item['category']}\n"
                f"Thương hiệu: {item.get('brand', 'Không rõ')}\n" # <-- Đã thêm Brand vào content
                f"Mô tả: {item['description']}\n"
                f"Giá: {item['price']:,} VNĐ\n"
                f"Đặc điểm: {', '.join(item['features'])}\n"
                f"Ảnh sản phẩm: {item.get('image_url', 'Không có ảnh')}"
            )
            metadata = {
                "id": item['id'],
                "name": item['name'],
                "category": item['category'],
                "brand": item.get('brand', ''), # <-- Đã thêm Brand vào metadata
                "price": item['price'],
                "image_url": item.get('image_url', '')
            }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def create_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

@tool
def add_to_cart(product_name: str, quantity: int = 1) -> str:
    """Adds a specified quantity of a product to the user's cart.
    Use this tool when the user explicitly asks to add a product to their cart.
    The product_name should be the exact name of the product the user is referring to.
    """
    global mock_cart

    found_product = None
    for p in all_products:
        if p["name"].lower() == product_name.lower():
            found_product = p
            break

    if found_product:
        if found_product["name"] in mock_cart:
            mock_cart[found_product["name"]] += quantity
        else:
            mock_cart[found_product["name"]] = quantity
        return f"Đã thêm {quantity} '{found_product['name']}' vào giỏ hàng của bạn. Giỏ hàng hiện tại: {mock_cart}"
    else:
        return f"Xin lỗi, tôi không tìm thấy sản phẩm '{product_name}' để thêm vào giỏ hàng."

@tool
def view_cart() -> str:
    """Views the current items in the user's shopping cart."""
    global mock_cart
    if not mock_cart:
        return "Giỏ hàng của bạn hiện đang trống."
    items = [f"{qty} x {name}" for name, qty in mock_cart.items()]
    return f"Giỏ hàng của bạn: {', '.join(items)}"

@tool
def calculate_cart_total() -> str:
    """Calculates the total price of all items currently in the user's shopping cart.
    Use this tool when the user asks about the total cost or checkout amount."""
    global mock_cart
    if not mock_cart:
        return "Giỏ hàng của bạn hiện đang trống, không có gì để tính tổng."

    total_amount = 0
    details = []

    product_dict = {p["name"].lower(): p for p in all_products}

    for item_name, quantity in mock_cart.items():
        product_info = product_dict.get(item_name.lower())
        if product_info:
            item_price = product_info["price"]
            subtotal = item_price * quantity
            total_amount += subtotal
            details.append(f"{quantity} x {item_name} ({item_price:,} VNĐ/cái) = {subtotal:,} VNĐ")
        else:
            details.append(f"{quantity} x {item_name} (Giá không xác định)")

    formatted_total = f"{total_amount:,} VNĐ"

    return f"Chi tiết giỏ hàng:\n" + "\n".join(details) + f"\nTổng cộng: **{formatted_total}**"

@tool
def update_cart_item(product_name: str, new_quantity: int) -> str:
    """Adjusts the quantity of an existing product in the shopping cart.
    Use this tool when the user wants to change the quantity of an item they already have in the cart.
    If new_quantity is 0, the item will be removed.
    """
    global mock_cart
    product_name_lower = product_name.lower()

    exact_product_name = None
    for item_in_cart in mock_cart.keys():
        if item_in_cart.lower() == product_name_lower:
            exact_product_name = item_in_cart
            break

    if exact_product_name:
        if new_quantity > 0:
            mock_cart[exact_product_name] = new_quantity
            return f"Đã cập nhật số lượng của '{exact_product_name}' thành {new_quantity} trong giỏ hàng."
        else:
            del mock_cart[exact_product_name]
            return f"Đã xóa '{exact_product_name}' khỏi giỏ hàng."
    else:
        return f"Sản phẩm '{product_name}' không có trong giỏ hàng để cập nhật."

@tool
def remove_from_cart(product_name: str) -> str:
    """Removes a specified product completely from the user's shopping cart.
    Use this tool when the user explicitly asks to remove an item from their cart.
    """
    global mock_cart
    product_name_lower = product_name.lower()

    exact_product_name = None
    for item_in_cart in mock_cart.keys():
        if item_in_cart.lower() == product_name_lower:
            exact_product_name = item_in_cart
            break

    if exact_product_name:
        del mock_cart[exact_product_name]
        return f"Đã xóa '{exact_product_name}' khỏi giỏ hàng của bạn."
    else:
        return f"Sản phẩm '{product_name}' không có trong giỏ hàng để xóa."

@tool
def proceed_to_checkout(payment_method: str = "general") -> str:
    """Initiates the checkout process for the items in the cart.
    This tool should be used when the user explicitly expresses intent to "checkout", "pay", or "proceed with payment".
    It will provide a simulated payment link and create a mock order.
    Args:
        payment_method (str): The preferred payment method (e.g., "paypal", "stripe", "visa"). Defaults to "general" if not specified.
    """
    global mock_cart, mock_orders, next_order_id

    if not mock_cart:
        return "Giỏ hàng của bạn hiện đang trống, không có gì để thanh toán."

    total_amount = 0
    order_items = []
    product_dict = {p["name"].lower(): p for p in all_products}

    for item_name, quantity in mock_cart.items():
        product_info = product_dict.get(item_name.lower())
        if product_info:
            item_price = product_info["price"]
            total_amount += item_price * quantity
            order_items.append({"name": item_name, "quantity": quantity, "price": item_price})

    formatted_total = f"{total_amount:,} VNĐ"

    payment_links = {
        "paypal": f"https://sandbox.paypal.com/checkout?amount={total_amount}&currency=VND&method=paypal",
        "stripe": f"https://checkout.stripe.com/pay/{total_amount}&currency=VND&method=stripe",
        "visa": f"https://secure.visa.com/checkout?amount={total_amount}&currency=VND&method=visa",
        "general": f"https://your-ecommerce-site.com/checkout?amount={total_amount}&currency=VND"
    }

    checkout_url = payment_links.get(payment_method.lower(), payment_links["general"])

    order_id = f"ORD{next_order_id:05d}"
    mock_orders[order_id] = {
        "items": order_items,
        "total_amount": total_amount,
        "payment_method": payment_method,
        "status": "Đang xử lý",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    next_order_id += 1

    mock_cart.clear()

    return (f"Tổng số tiền cần thanh toán là **{formatted_total}**. "
            f"Vui lòng nhấn vào liên kết sau để hoàn tất thanh toán bằng {payment_method.upper()}:\n"
            f"🔗 [Thanh toán ngay]({checkout_url})\n"
            f"Đơn hàng của bạn đã được tạo với mã **{order_id}**. Chúng tôi sẽ gửi xác nhận qua email. Cảm ơn bạn đã mua sắm!")

@tool
def filter_products(
    category: str = None,
    min_price: int = None,
    max_price: int = None,
    features: str = None,
    name: str = None,
    brand: str = None # <-- Đã thêm đối số brand
) -> str:
    """
    Filters products based on specified criteria such as category, price range, features, exact name, or brand.
    Use this tool when the user asks to find products with specific conditions.
    Args:
        category (str): The category of products to filter (e.g., "Điện thoại", "Laptop", "Tai nghe").
        min_price (int): The minimum price of the products.
        max_price (int): The maximum price of the products.
        features (str): A comma-separated string of features to look for (e.g., "Pin trâu, Camera chất lượng cao").
        name (str): An exact product name to search for.
        brand (str): The brand of the products (e.g., "Samsung", "Apple", "Dell"). # <-- Mô tả brand mới
    Returns:
        A string listing matching products with their basic info.
    """

    matching_products = []

    for product in all_products:
        match = True

        if category and product["category"].lower() != category.lower():
            match = False

        if min_price is not None and product["price"] < min_price:
            match = False

        if max_price is not None and product["price"] > max_price:
            match = False

        if features:
            required_features = [f.strip().lower() for f in features.split(',')]
            product_features_lower = [f.lower() for f in product["features"]]
            if not all(rf in product_features_lower for rf in required_features):
                match = False

        if name and product["name"].lower() != name.lower():
            match = False

        # LOGIC LỌC THEO THƯƠNG HIỆU MỚI
        if brand and product.get("brand", "").lower() != brand.lower():
            match = False

        if match:
            matching_products.append(product)

    if not matching_products:
        return "Xin lỗi, không tìm thấy sản phẩm nào phù hợp với tiêu chí bạn đưa ra."

    results = ["Dưới đây là các sản phẩm phù hợp:"]
    for p in matching_products:
        results.append(
            f"- **{p['name']}** ({p['category']}, {p.get('brand', 'Không rõ')}) - Giá: {p['price']:,} VNĐ. Đặc điểm: {', '.join(p['features'])}. Ảnh: {p.get('image_url', 'Không có')}"
        )
    return "\n".join(results)

@tool
def get_order_status(order_id: str) -> str:
    """
    Retrieves the current status and details of a specific order using its order ID.
    Use this tool when the user asks about the status of their order or wants to view a specific order.
    """
    global mock_orders
    order = mock_orders.get(order_id)
    if order:
        items_str = ", ".join([f"{item['quantity']}x {item['name']}" for item in order['items']])
        return (f"Đơn hàng **{order_id}** của bạn:\n"
                f"- Trạng thái: **{order['status']}**\n"
                f"- Tổng tiền: {order['total_amount']:,} VNĐ\n"
                f"- Phương thức thanh toán: {order['payment_method'].upper()}\n"
                f"- Sản phẩm: {items_str}\n"
                f"- Thời gian đặt: {order['timestamp']}")
    else:
        return f"Xin lỗi, không tìm thấy đơn hàng với mã **{order_id}**."

@tool
def get_all_orders() -> str:
    """
    Retrieves a list of all past orders.
    Use this tool when the user asks to see their order history or all placed orders.
    """
    global mock_orders
    if not mock_orders:
        return "Bạn chưa có đơn hàng nào."

    order_list = ["Danh sách các đơn hàng của bạn:"]
    for order_id, order in mock_orders.items():
        order_list.append(f"- **{order_id}**: {order['total_amount']:,} VNĐ, Trạng thái: **{order['status']}** ({order['timestamp']})")

    return "\n".join(order_list)

@tool
def recommend_products(based_on_product: str = None) -> str:
    """Recommends other products that might be of interest to the user,
    based on a specified product or their general Browse history.
    Use this tool when the user expresses general interest in products,
    asks for suggestions, or after discussing a specific product.
    Args:
        based_on_product (str): An optional product name to base recommendations on.
                                If not provided, general popular products or
                                items from user_product_interest_history will be used.
    """
    global all_products, user_product_interest_history

    if not all_products:
        return "Xin lỗi, hiện tại không có sản phẩm nào để gợi ý."

    recommended_products = []
    seen_ids = set() # Dùng để tránh gợi ý trùng lặp

    # Ưu tiên gợi ý dựa trên sản phẩm cụ thể nếu có
    if based_on_product:
        found_product = next((p for p in all_products if p["name"].lower() == based_on_product.lower()), None)
        if found_product:
            # Gợi ý sản phẩm cùng danh mục hoặc có tính năng tương tự
            for p in all_products:
                # Đảm bảo không gợi ý lại chính sản phẩm đó và đã chưa được thêm vào
                if p["id"] != found_product["id"] and p["category"] == found_product["category"] and p["id"] not in seen_ids:
                    recommended_products.append(f"- {p['name']} ({p['category']}) - Giá: {p['price']:,} VNĐ")
                    seen_ids.add(p['id'])
                    if len(recommended_products) >= 3: # Giới hạn 3 gợi ý ban đầu
                        break
            # Nếu không có sản phẩm cùng danh mục, thử các sản phẩm nổi bật khác
            if not recommended_products or len(recommended_products) < 3:
                 for p in all_products:
                    if p["id"] != found_product["id"] and p["id"] not in seen_ids:
                        recommended_products.append(f"- {p['name']} ({p['category']}) - Giá: {p['price']:,} VNĐ")
                        seen_ids.add(p['id'])
                        if len(recommended_products) >= 3:
                            break

    # Nếu không có sản phẩm cụ thể hoặc chưa đủ gợi ý, dùng lịch sử quan tâm của người dùng
    if not recommended_products and user_product_interest_history:
        # Lấy sản phẩm gần đây nhất mà người dùng quan tâm
        latest_interest = user_product_interest_history[-1] if user_product_interest_history else None
        if latest_interest:
            found_product = next((p for p in all_products if p["name"].lower() == latest_interest.lower()), None)
            if found_product:
                for p in all_products:
                    if p["id"] != found_product["id"] and p["category"] == found_product["category"] and p["id"] not in seen_ids:
                        recommended_products.append(f"- {p['name']} ({p['category']}) - Giá: {p['price']:,} VNĐ")
                        seen_ids.add(p['id'])
                        if len(recommended_products) >= 3:
                            break

    # Nếu vẫn chưa có hoặc ít gợi ý, lấy ngẫu nhiên các sản phẩm nổi bật (ví dụ: top 3 sản phẩm)
    if len(recommended_products) < 3:
        for p in all_products:
            if p["id"] not in seen_ids:
                recommended_products.append(f"- {p['name']} ({p['category']}) - Giá: {p['price']:,} VNĐ")
                seen_ids.add(p['id'])
                if len(recommended_products) >= 3:
                    break

    if recommended_products:
        return "Có vẻ bạn quan tâm đến các sản phẩm này. Bạn có thể tham khảo thêm:\n" + "\n".join(recommended_products)
    else:
        return "Hiện tại tôi chưa có gợi ý cụ thể nào cho bạn. Bạn có muốn xem danh mục sản phẩm nào không?"


def setup_chatbot(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một trợ lý bán hàng thân thiện và hữu ích. Bạn sẽ trả lời các câu hỏi của người dùng dựa trên thông tin được cung cấp. Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không có thông tin đó và đề xuất các chủ đề khác mà bạn có thể hỗ trợ (ví dụ: chính sách bảo hành, đổi trả, thời gian giao hàng)."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Thông tin liên quan:\n{context}")
    ])

    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

    history_aware_retriever = create_history_aware_retriever(llm, retriever, ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", "Dựa trên lịch sử trò chuyện trên và câu hỏi mới của người dùng, tạo một truy vấn tìm kiếm phù hợp để tìm kiếm thông tin về sản phẩm, chính sách, hoặc thông tin chung của cửa hàng. Nếu câu hỏi không liên quan đến tìm kiếm thông tin, hãy bỏ qua."),
    ]))

    rag_chain_for_retrieval = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    @tool
    def retrieve_info(query: str) -> str:
        """Use this tool to answer any general information questions, including details about products,
        store policies (like return or warranty), shop address, contact information for support,
        and delivery times. Provide the user's exact question as the 'query'.
        Returns relevant informational details from the knowledge base.
        IMPORTANT: After using this tool to describe a specific product, check if the product name
        can be added to `user_product_interest_history` for later recommendations."""
        global user_product_interest_history # Cần global để GHI vào danh sách

        response = rag_chain_for_retrieval.invoke({"input": query, "chat_history": []})
        answer = response["answer"]

        # Trích xuất tên sản phẩm từ câu trả lời để lưu vào lịch sử quan tâm
        for product in all_products:
            # Kiểm tra xem tên sản phẩm có trong câu trả lời không, không phân biệt hoa thường
            if product["name"].lower() in answer.lower():
                if product["name"] not in user_product_interest_history:
                    user_product_interest_history.append(product["name"])
                break
        return answer

    tools = [
        add_to_cart,
        view_cart,
        calculate_cart_total,
        update_cart_item,
        remove_from_cart,
        proceed_to_checkout,
        filter_products,
        get_order_status,
        get_all_orders,
        retrieve_info, # Công cụ tra cứu thông tin chung
        recommend_products # Công cụ gợi ý sản phẩm mới
    ]

    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một trợ lý bán hàng chuyên nghiệp và hỗ trợ khách hàng toàn diện. Bạn có thể trả lời các câu hỏi về sản phẩm (bao gồm tìm kiếm chi tiết theo thuộc tính), chính sách cửa hàng (đổi trả, bảo hành), địa chỉ cửa hàng, thông tin liên hệ, thời gian giao hàng, giúp người dùng thực hiện các hành động liên quan đến giỏ hàng (thêm, xem, cập nhật số lượng, xóa sản phẩm), tính tổng tiền giỏ hàng, hỗ trợ thanh toán, và quản lý đơn hàng (xem trạng thái, lịch sử)."
                   "Nếu cần, hãy sử dụng các công cụ có sẵn. Luôn thân thiện và hữu ích. "
                   "**CHỈ SỬ DỤNG CÔNG CỤ 'retrieve_info' cho TẤT CẢ các câu hỏi liên quan đến thông tin chung, bao gồm: mô tả sản phẩm, chi tiết về chính sách (đổi trả, bảo hành), địa chỉ cửa hàng, cách liên hệ hỗ trợ, hoặc thời gian giao hàng. Đảm bảo trích xuất chính xác câu hỏi của người dùng để truyền vào công cụ này.**"
                   "**SỬ DỤNG CÔNG CỤ 'filter_products' khi người dùng muốn tìm sản phẩm với các TIÊU CHÍ LỌC CỤ THỂ như 'điện thoại dưới 10 triệu', 'laptop RAM 16GB', 'tai nghe chống ồn', 'điện thoại Samsung' hoặc 'điện thoại của Apple', và đảm bảo truyền đúng các đối số (category, min_price, max_price, features, name, brand).**" # <-- Cập nhật hướng dẫn cho brand
                   "**SỬ DỤNG CÔNG CỤ 'recommend_products' khi người dùng HỎI GỢI Ý SẢN PHẨM, nói 'tôi muốn xem thêm', 'có gì hay không', hoặc khi bạn đã trả lời một câu hỏi về sản phẩm nhưng không có hành động cụ thể nào khác. Nếu có sản phẩm nào đó vừa được thảo luận, hãy truyền tên sản phẩm đó vào đối số 'based_on_product' để gợi ý liên quan.**"
                   "**SỬ DỤNG CÔNG CÔNG CỤ 'add_to_cart' khi người dùng muốn THÊM sản phẩm vào giỏ hàng.**"
                   "**SỬ DỤNG CÔNG CỤ 'calculate_cart_total' khi người dùng hỏi về tổng tiền giỏ hàng hoặc số tiền cần thanh toán.** "
                   "**SỬ DỤNG CÔNG CỤ 'update_cart_item' khi người dùng muốn THAY ĐỔI SỐ LƯỢNG sản phẩm trong giỏ hàng (ví dụ: 'thay đổi số lượng iPhone thành 3'). Nếu số lượng mới là 0, hãy hiểu là xóa sản phẩm.** "
                   "**SỬ DỤNG CÔNG CỤ 'remove_from_cart' khi người dùng muốn XÓA HOÀN TOÀN một sản phẩm khỏi giỏ hàng (ví dụ: 'xóa iPhone khỏi giỏ').**"
                   "**SỬ DỤNG CÔNG CỤ 'view_cart' khi người dùng muốn xem các mặt hàng hiện tại trong giỏ hàng.**"
                   "**SỬ DỤNG CÔNG CỤ 'proceed_to_checkout' khi người dùng muốn TIẾN HÀNH THANH TOÁN, MUA HÀNG, hoặc HOÀN TẤT ĐƠN HÀNG. Truyền 'payment_method' nếu người dùng chỉ định.**"
                   "**SỬ DỤNG CÔNG CỤ 'get_order_status' khi người dùng hỏi về TRẠNG THÁI của một đơn hàng cụ thể (ví dụ: 'Trạng thái đơn hàng ORD00001 là gì?').**"
                   "**SỬ DỤNG CÔNG CỤ 'get_all_orders' khi người dùng muốn XEM TẤT CẢ các đơn hàng đã đặt hoặc lịch sử mua hàng.**"
                   "Luôn cung cấp thông tin hữu ích và hướng dẫn rõ ràng cho người dùng."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return rag_chain_for_retrieval, agent_executor

def get_response(user_query: str, chat_history: list, agent_executor) -> str:
    response = agent_executor.invoke({"input": user_query, "chat_history": chat_history})
    return response["output"]

if __name__ == "__main__":
    print("Đang khởi tạo chatbot. Vui lòng chờ...")

    product_documents = load_products_data_from_api()

    vectorstore = create_vector_store(product_documents)

    _, agent_executor = setup_chatbot(vectorstore)
    print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")

    chat_history = []

    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'exit':
            print("Tạm biệt!")
            break

        response = get_response(user_input, chat_history, agent_executor)
        print(f"Bot: {response}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))