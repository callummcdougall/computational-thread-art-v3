import base64
import gc
import io
import os
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

from PIL import Image

import streamlit as st

# https://info.snowflake.com/streamlit-resource-increase-request.html?ref=blog.streamlit.io


# Set page configuration
st.set_page_config(page_title="Thread Art Generator", page_icon="🧵", layout="wide", initial_sidebar_state="expanded")

# Add parent directory to path so we can import the required modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# for path in Path(parent_dir).iterdir():
#     st.write(path)
#     if not path.is_file():
#         for subpath in path.iterdir():
#             st.write("sub: ", subpath)

os.chdir(parent_dir)

gc.collect()

from image_color import Img, ThreadArtColorParams
from streamlit.components.v1 import html as st_html

# Apply custom CSS for a clean, minimalist look
st.markdown(
    """
<style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        font-weight: 400;
    }
    .stButton button {
        width: 100%;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 0.5rem;
    }
    .color-box {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 10px;
        border: 1px solid #ccc;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("Thread Art Generator")
st.markdown("Create beautiful thread art from images with customizable parameters.")
st.markdown("You can upload your own image and select parameters using the menu on the left.")
st.markdown(
    "You can also choose a demo image to set the image & parameters, and scroll down to the 'Generate Thread Art' button to get started right away!"
)

# Initialize session state
if "generated_html" not in st.session_state:
    st.session_state.generated_html = None
if "output_name" not in st.session_state:
    st.session_state.output_name = None
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.TemporaryDirectory()

name = None

# parameters
demo_presets = {
    "Custom": {},
    "Tiger Demo (fast)": {
        "filename": "tiger.jpg",
        "name": "tiger_small_01",
        "x": 600,
        "nodes": 320,
        "shape": "Rectangle",
        "random_lines": 140,
        "darkness": {
            "white": 0.16,
            "orange": 0.16,
            "red": 0.16,
            "black": 0.16,
        },
        "blur": 4,
        "group_orders": "5",
        "palette": {
            "white": [255, 255, 255],
            "orange": [255, 130, 0],
            "red": [255, 0, 0],
            "black": [0, 0, 0],
        },
        "lines": [2700, 2000, 650, 5200],
        "html_x": 700,
    },
    "Tiger Demo (slow)": {
        "filename": "tiger.jpg",
        "name": "tiger_big_01",
        "x": 700,
        "nodes": 400,
        "shape": "Rectangle",
        "random_lines": 200,
        "darkness": {
            "black": 0.12,
            "white": 0.12,
            "orange": 0.12,
            "red": 0.12,
        },
        "blur": 4,
        "group_orders": "8",
        "palette": {
            "white": [255, 255, 255],
            "orange": [255, 130, 0],
            "red": [255, 0, 0],
            "black": [0, 0, 0],
        },
        "lines": [5400, 4000, 2000, 9500],
        "html_x": 800,
        "html_line_width": 0.11,
    },
    "Stag Demo (fast)": {
        "filename": "stag-large.jpg",
        "name": "stag_small_01",
        "x": 1200,
        "nodes": 360,
        "shape": "Rectangle",
        "random_lines": 180,
        "darkness": {
            "white": 0.14,
            "light_blue": 0.14,
            "mid_blue": 0.10,
            "dark_blue": 0.11,
            "black": 0.10,
        },
        "blur": 4,
        "group_orders": "wdlbwdmlbwdmlbmb",
        "palette": {
            "white": [255, 255, 255],
            "light_blue": [0, 215, 225],
            "mid_blue": [0, 120, 240],
            "dark_blue": [0, 0, 120],
            "black": [0, 0, 0],
        },
        "lines": [1400, 750, 750, 3000, 6500],
        "html_x": 1100,
    },
    "Stag Demo (slow)": {
        "filename": "stag.jpg",
        "name": "stag_large_01",
        "x": 1400,
        "nodes": 400,
        "shape": "Rectangle",
        "random_lines": 240,
        "darkness": {
            "white": 0.14,
            "light_blue": 0.13,
            "mid_blue": 0.10,
            "dark_blue": 0.10,
            "black": 0.10,
        },
        "blur": 4,
        "group_orders": "wdlbwdmlbwdmlbmb",
        "palette": {
            "white": [255, 255, 255],
            "light_blue": [0, 215, 225],
            "mid_blue": [0, 120, 240],
            "dark_blue": [0, 0, 120],
            "black": [0, 0, 0],
        },
        "lines": [1600, 900, 850, 3300, 8000],
        "html_x": 1200,
        "html_line_width": 0.11,
    },
    "Duck Demo": {
        "filename": "duck.jpg",
        "name": "duck_01",
        "x": 660,
        "nodes": 360,
        "shape": "Rectangle",
        "random_lines": 150,
        "darkness": {
            "white": 0.12,
            "yellow": 0.12,
            "red": 0.12,
            "black": 0.12,
        },
        "blur": 4,
        "group_orders": "wrywrybwrybrybrbb",
        "palette": {
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "yellow": [255, 255, 0],
            "black": [0, 0, 0],
        },
        "lines": [1800, 800, 1800, 8000],
        "html_x": 1000,
    },
    "Fish Demo": {
        "filename": "fish_sq_2.jpg",
        "name": "fish_01",
        "x": 1100,
        "nodes": 360,
        "shape": "Ellipse",
        "random_lines": 200,
        "darkness": {
            "white": 0.28,
            "orange": 0.25,
            "mid_blue": 0.28,
            "black": 0.25,
        },
        "blur": 4,
        "group_orders": "wwoombwombb",
        "palette": {
            "white": [255, 255, 255],
            "orange": [255, 100, 0],
            "mid_blue": [50, 150, 220],
            "black": [0, 0, 0],
        },
        "lines": [500, 1900, 2100, 3300],
        "html_x": 850,
    },
    "Snake Demo": {
        "filename": "snake.png",
        "name": "snake_01",
        "x": 1200,
        "nodes": 360,
        "shape": "Rectangle",
        "random_lines": 160,
        "darkness": {"white": 0.12, "yellow": 0.12, "red": 0.12, "black": 0.14},
        "blur": 2,
        "group_orders": "wwyyrrbwyrbwyrbwyrbbb",
        "palette": {
            "white": [255, 255, 255],
            "yellow": [255, 255, 0],
            "red": [255, 0, 0],
            "black": [0, 0, 0],
        },
        "lines": [1300, 1500, 1200, 11500],
        "html_x": 1000,
    },
    "Planets Demo": {
        "filename": "planets-1-Ga.png",
        "w_filename": "planets-1-GwA.png",
        "name": "planets_01",
        "x": 1200,
        "nodes": 360,
        "shape": "Rectangle",
        "random_lines": 180,
        "darkness": {
            "white": 0.21,
            "yellow": 0.21,
            "mid_blue": 0.21,
            "red": 0.21,
            "dark_brown": 0.21,
            "black": 0.14,
        },
        "blur": 2,
        "group_orders": "wwwyyyrrmmwyrrmdbymrddbbb",
        "palette": {
            "white": [255, 255, 255],
            "yellow": [230, 200, 80],
            "mid_blue": [0, 50, 200],
            "red": [255, 0, 0],
            "dark_brown": [140, 60, 0],
            "black": [0, 0, 0],
        },
        "lines": [750, 900, 500, 450, 1200, 9200],
        "html_x": 1200,
    },
}

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters")

    images = {
        "uploaded": None,
        "demo": None,
    }

    # We need to reset stored HTML when this changes
    def reset():
        st.session_state.generated_html = None
        st.session_state.output_name = None
        st.session_state.sf = None

    # Demo selector
    demo_option = st.selectbox(
        "Choose a demo or create your own",
        demo_presets.keys(),
        help="Select a demo to try out the thread art generator with preset parameters. Some of the demos are labelled with (fast) or (long) to indicate how long they will take to generate - the (long) images are larger and more detailed.",
        on_change=reset,
    )

    preset_filename = demo_presets[demo_option].get("filename", None)
    preset_w_filename = demo_presets[demo_option].get("w_filename", None)
    preset_name = demo_presets[demo_option].get("name", None)
    preset_x = demo_presets[demo_option].get("x", None)
    preset_html_x = demo_presets[demo_option].get("html_x", None)
    preset_html_line_width = demo_presets[demo_option].get("html_line_width", None)
    preset_nodes = demo_presets[demo_option].get("nodes", None)
    preset_shape = demo_presets[demo_option].get("shape", None)
    preset_random_lines = demo_presets[demo_option].get("random_lines", None)
    preset_darkness = demo_presets[demo_option].get("darkness", None)
    preset_blur = demo_presets[demo_option].get("blur", None)
    preset_group_orders = demo_presets[demo_option].get("group_orders", None)
    preset_palette = demo_presets[demo_option].get("palette", None)
    preset_darkness = demo_presets[demo_option].get("darkness", None)
    preset_lines = demo_presets[demo_option].get("lines", None)
    preset_step_size = demo_presets[demo_option].get("step_size", None)

    image_selected = False
    image = None

    if demo_option == "Custom":
        # User uploads their own image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            image_selected = True
    else:
        # Read the image from `images/` and display it
        demo_image_path = Path(__file__).parent.parent / "images" / preset_filename
        with demo_image_path.open("rb") as f:
            image_bytes = f.read()
            image_selected = True

    if image_selected:
        image = Image.open(io.BytesIO(image_bytes))
        st.image(
            image,
            use_container_width=True,
        )

    # Basic parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        x_size = st.number_input(
            "Width",
            min_value=100,
            max_value=1400,
            value=preset_x or 600,
        )
    with col2:
        n_nodes = st.number_input(
            "Number of Nodes",
            min_value=60,
            max_value=400,
            value=preset_nodes or 320,
            step=20,
            help="Number of nodes on the perimeter of the image to generate lines between. This increases resolution but also time to create the image.",
        )
        n_nodes_real = n_nodes + (4 - n_nodes % 4)  # Ensure n_nodes is a multiple of 4
    with col3:
        shape = st.selectbox(
            "Shape",
            ["Rectangle", "Ellipse"],
            index=0 if preset_shape == "Rectangle" or preset_shape is None else 1,
            help="Options are Rectangle or Ellipse. If Ellipse then the nodes we generate lines between will be placed in an ellipse shape (cropping the image appropriately).",
        )

    # Advanced parameters
    with st.expander("Advanced Parameters"):
        n_random_lines = st.number_input(
            "Random Lines to Consider",
            min_value=10,
            max_value=500,
            value=preset_random_lines or 150,
            help="Number of random lines to consider each time we draw a new line. More lines takes longer, but leads to a higher resolution image (although past about 150 you get diminishing returns).",
        )

        blur_rad = st.number_input(
            "Blur Radius",
            min_value=0,
            max_value=20,
            value=preset_blur or 4,
            help="Amount we blur the monochrome images when we split them off from the main image. This usually doesn't matter much (but it can help to increase it if the lines seem too sharp and you want the color gradients to be smoother).",
        )

        group_orders = st.text_input(
            "Group Orders",
            value=preset_group_orders or "4",
            help="""Sequence of first letters of each color, repeated. For example, 'orborb' means we draw the image in the following order: 50% orange, 50% red, 50% black, 50% orange, 50% red, 50% black lines. We recommend about 4 cycles through the colors, with darker colors coming last (so the final lines drawn on top of the image are black).
            
You can optionally just put a number, in which case it'll cycle through all the colors from lightest to darkest that many times.""",
        )

    # Color management
    st.subheader("Colors")

    if preset_palette:
        colors = list(preset_palette.keys())
        color_values = list(preset_palette.values())
        n_lines = preset_lines
        darkness_values = list(preset_darkness.values())
    else:
        # Default to 3 colors if custom
        colors = ["black", "red", "white"]
        color_values = [[0, 0, 0], [255, 0, 0], [255, 255, 255]]
        n_lines = [1000, 800, 600]
        darkness_values = [0.17, 0.17, 0.17]

    # Allow adding or removing colors
    num_colors = st.number_input(
        "Number of Colors",
        min_value=1,
        max_value=10,
        value=len(colors),
        help="We recommend always including black and white, as well as between 1 and 4 other colors depending on your image. There should usually be a larger number of darker colored lines than most other colors (but again this depends on the image).",
    )

    if num_colors != len(colors):
        if num_colors > len(colors):
            # Add more colors
            for i in range(len(colors), num_colors):
                colors.append(f"color_{i + 1}")
                color_values.append([128, 128, 128])  # Default to gray
                n_lines.append(1000)  # Default number of lines
                darkness_values.append(0.17)
        else:
            # Remove colors
            colors = colors[:num_colors]
            color_values = color_values[:num_colors]
            n_lines = n_lines[:num_colors]
            darkness_values = darkness_values[:num_colors]

    # Color editors
    new_colors = []
    new_color_values = []
    new_n_lines = []
    new_darkness = []

    for i in range(num_colors):
        # st.markdown(f"##### Color {i + 1}")
        col1, col2, col3, col4 = st.columns([3, 2, 3, 4])

        with col1:
            color_name = st.text_input(
                "Name",
                value=colors[i],
                key=f"color_name_{i}",
                help="Make sure to give the colors names that don't start with the same letter (otherwise later code will get confused).",
            )
            # TODO - fix this behaviour

        with col2:
            color_hex = st.color_picker(
                "Color",
                f"#{color_values[i][0]:02x}{color_values[i][1]:02x}{color_values[i][2]:02x}",
                key=f"color_pick_{i}",
            )
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)

        with col3:
            lines = st.number_input(
                "Lines",
                min_value=100,
                max_value=15000,
                value=n_lines[i],
                key=f"lines_{i}",
                help="The total number of lines we'll draw for this color. Make sure this is larger for the darker colors, but other than that it should roughly be in proportion with the color density in your reference image.",
            )

        with col4:
            darkness = st.number_input(
                "Darkness",
                min_value=0.05,
                max_value=0.3,
                value=darkness_values[i],
                key=f"darkness_{i}",
                step=0.01,
                help="The float value we'll subtract from pixels after each line is drawn (maximum-darkness pixels start at the value of 1.0). Smaller values here mean higher contrast (because we put more lines in the dark areas before moving to the light areas).",
            )

        new_colors.append(color_name)
        new_color_values.append([r, g, b])
        new_n_lines.append(lines)
        new_darkness.append(darkness)

    # Update colors and lines
    colors = new_colors
    color_values = new_color_values
    n_lines = new_n_lines
    darkness = {colors[i]: new_darkness[i] for i in range(len(colors))}

    # Check if colors have the same first letter; throw a warning if they do
    first_letters = [color[0] for color in colors]
    if len(first_letters) != len(set(first_letters)):
        st.error("Warning: some color names have the same first letter. Please ensure all color names are unique.")

    # HTML output options
    st.subheader("Output")

    cols = st.columns(2)
    with cols[0]:
        html_line_width = st.number_input(
            "Line width (output)",
            min_value=0.05,
            max_value=0.3,
            value=preset_html_line_width or 0.13,
            step=0.01,
            help="Width of the lines in the output image. Generally this can be kept at 0.14; smaller values mean thinner lines and look better when your images are very large and have a lot of lines.",
        )
    with cols[1]:
        html_width = st.number_input(
            "Image width (output)",
            min_value=300,
            max_value=2000,
            value=preset_html_x or preset_x or 800,
            step=50,
            help="Width of the output image in pixels. Increasing this will mean the final image takes longer to generate, but looks higher-resolution.",
        )

    # Generate button
    generate_button = st.button("Generate Thread Art", type="primary")

# Process the image and generate thread art
if generate_button:
    if not image_selected:
        st.error("Please upload an image or select a demo.")
        st.stop()

    name = preset_name or "custom_thread_art"

    # if isinstance(demo_image_path, Path) and demo_image_path.exists():
    #     image_path = demo_image_path.name
    #     w_filename = None
    # elif uploaded_file is not None:
    #     # Save the uploaded file to a temporary location
    #     temp_img = Path(st.session_state.temp_dir.name) / f"uploaded_image.{uploaded_file.name.split('.')[-1]}"
    #     with open(temp_img, "wb") as f:
    #         f.write(uploaded_file.getbuffer())
    #     image_path = temp_img.name
    #     w_filename = None
    # else:
    #     st.error("Please upload an image or select a demo.")
    #     st.stop()

    # Create palette dictionary
    palette = {colors[i]: color_values[i] for i in range(len(colors))}

    # Display a status message
    try:
        with st.spinner("Preprocessing (takes about 10-20 seconds) ..."):
            # Set up parameters
            args = ThreadArtColorParams(
                name=name,
                x=x_size,
                n_nodes=n_nodes_real,
                filename=None,
                w_filename=preset_w_filename,
                palette=palette,
                n_lines_per_color=n_lines,
                shape=shape,
                n_random_lines=n_random_lines,
                darkness=darkness,
                blur_rad=blur_rad,
                group_orders=group_orders,
                image=image,
                step_size=preset_step_size or 1.618,  # golden ratio for the lulz
            )

            # Create image object
            my_img = Img(args)

        # Get the line dictionary (using progress bar)
        line_dict = defaultdict(list)
        total_lines = sum(my_img.args.n_lines_per_color)
        progress_bar = st.progress(0, text="Generating lines...")
        progress_count = 0
        for color, i, j in my_img.create_canvas_generator():
            line_dict[color].append((i, j))
            progress_count += 1
            progress_bar.progress(progress_count / total_lines, text="Generating lines...")

        # Generate HTML
        html_content = my_img.generate_thread_art_html(
            line_dict,
            x=html_width,
            line_width=html_line_width,
            steps_per_slider=150,
            rand_perm=0.0025,
            bg_color=(0, 0, 0),
        )

        # Success message
        st.success("Thread art generated successfully!")

        # Store the generated HTML, and delete what we don't need any more
        st.session_state.generated_html = html_content
        st.session_state.sf = my_img.y / my_img.x

        del args
        del my_img
        del line_dict
        gc.collect()

    except Exception as e:
        st.error(f"Error generating thread art: {str(e)}")
        st.code(traceback.format_exc())


# Display the generated thread art if available
if st.session_state.generated_html:
    st.header("Generated Thread Art")

    # Display the HTML output
    html_height = html_width * st.session_state.sf
    st_html(st.session_state.generated_html, height=html_height + 150, scrolling=True)

    # Download options
    st.subheader("Download Options")

    # Provide HTML download
    b64_html = base64.b64encode(st.session_state.generated_html.encode()).decode()
    href_html = f'<a href="data:text/html;base64,{b64_html}" download="{name}.html">Download HTML File</a>'
    st.markdown(href_html, unsafe_allow_html=True)

    # # Show embed code for Squarespace
    # st.subheader("Embed Code for Squarespace")
    # st.text_area("Copy this code into a Code Block in Squarespace:", st.session_state.generated_html, height=200)
    # st.markdown("""
    # Instructions:
    # 1. Copy the code above
    # 2. In Squarespace, add a "Code" block to your page
    # 3. Paste the code into the block
    # 4. Save your changes
    # """)

# netstat -ano | findstr "0.0.0.0:8501.*LISTENING"
# psrecord 49256 --plot plot.png

# # simpler / longer versions:
# netstat -ano | findstr :8501
# netstat -ano | findstr ":8501.*LISTENING" | for /f "tokens=5" %a in ('findstr /i "listening"') do @echo %a
