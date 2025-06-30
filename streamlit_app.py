import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time
import re # Import for regular expressions to extract product names

# Import necessary components from chatbot_core
# Note: all_products and mock_cart are globally defined in chatbot_core.py
# and we can import them directly to share state.
from chatbot_core import setup_chatbot, load_products_data_from_api, get_response, mock_cart, all_products, create_vector_store

# --- Cấu hình trang Streamlit ---
st.set_page_config(page_title="E-commerce Chatbot", page_icon="🛍️", layout="centered") # layout centered for better look
st.title("🛍️ Chatbot của bạn")
st.caption("Trò chuyện để tìm sản phẩm, thêm vào giỏ hàng, hoặc xem tổng tiền.")

# --- Khởi tạo Chatbot (Chỉ chạy một lần) ---
if "agent_executor" not in st.session_state:
    st.write("Đang khởi tạo chatbot. Vui lòng chờ...")
    with st.spinner("Đang tải dữ liệu và mô hình..."):
        try:
            product_documents = load_products_data_from_api() 
            vectorstore = create_vector_store(product_documents)
            _, agent_executor = setup_chatbot(vectorstore)
            st.session_state.agent_executor = agent_executor
            st.success("Chatbot đã sẵn sàng!")
        except Exception as e:
            st.error(f"Lỗi khi khởi tạo chatbot: {e}")
            st.stop() 

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Chào bạn! Tôi có thể giúp gì cho bạn hôm nay? (Ví dụ: 'Tìm điện thoại màn hình lớn', 'Thêm iPhone 15 Pro Max vào giỏ hàng', 'Xem giỏ hàng', 'Tính tổng tiền giỏ hàng')")
    ]

# --- Hiển thị lịch sử trò chuyện ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # If the message contains product info, try to display image
            # This is a heuristic and might need refinement for complex answers
            product_found_in_response = False
            for product in all_products:
                if product["name"].lower() in message.content.lower() and product.get("image_url"):
                    st.image(product["image_url"], caption=product["name"], width=150)
                    product_found_in_response = True
                    break
            st.markdown(message.content)


# --- Xử lý đầu vào từ người dùng ---
user_query = st.chat_input("Nhập tin nhắn của bạn...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Bot đang suy nghĩ..."):
            try:
                response = get_response(user_query, st.session_state.chat_history, st.session_state.agent_executor)
                
                # Check for product image in the response
                # This part is key for displaying images
                product_image_displayed = False
                for product in all_products:
                    # Simple heuristic: if product name is in response AND it has an image URL
                    if product["name"].lower() in response.lower() and product.get("image_url"):
                        st.image(product["image_url"], caption=product["name"], width=150)
                        product_image_displayed = True
                        break # Only display the first matching product image

                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi xử lý yêu cầu của bạn: {e}")
                st.session_state.chat_history.append(AIMessage(content="Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau."))

# --- Hiển thị giỏ hàng hiện tại ---
st.sidebar.title("🛒 Giỏ hàng của bạn")
if not mock_cart:
    st.sidebar.markdown("_Giỏ hàng trống._")
else:
    total_sidebar = 0
    product_dict_sidebar = {p["name"].lower(): p for p in all_products} # For quick lookup
    for product_name, qty in mock_cart.items():
        st.sidebar.write(f"- **{qty}** x {product_name}")
        product_info = product_dict_sidebar.get(product_name.lower())
        if product_info:
            item_price = product_info["price"]
            subtotal = item_price * qty
            total_sidebar += subtotal
            st.sidebar.caption(f"  _({item_price:,} VNĐ/cái)_")
    st.sidebar.markdown(f"---")
    st.sidebar.markdown(f"**Tổng cộng:** **{total_sidebar:,} VNĐ**")

# You might still see telemetry warnings from chromadb in the console, they are harmless.