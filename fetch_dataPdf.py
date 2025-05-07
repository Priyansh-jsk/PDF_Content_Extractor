import streamlit as st
import fitz
import io
from PIL import Image
import pandas as pd
import cohere
import numpy as np
# from tempfile import NamedTemporaryFile

st.set_page_config(page_title="PDF Content Extractor", layout="wide")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF."""
    text = ""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def extract_images_from_pdf(pdf_file):
    """Extract images from PDF using PyMuPDF."""
    images = []
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append((f"Page {page_num+1}, Image {img_index+1}", image))
                
        return images
    except Exception as e:
        st.error(f"Error extracting images: {e}")
        return []

def extract_tables_from_pdf_pymupdf(pdf_file):
    """Extract tables from PDF using PyMuPDF's built-in table detection."""
    tables = []
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Extract tables using PyMuPDF's built-in table detection
            tabs = page.find_tables()
            for i, table in enumerate(tabs.tables):
                # Convert PyMuPDF table to pandas DataFrame
                rows_data = []
                for row in range(table.rows):
                    row_data = []
                    for col in range(table.cols):
                        row_data.append(table.cell(row, col).text)
                    rows_data.append(row_data)
                
                # Create column names (either first row or generic)
                if len(rows_data) > 0:
                    if all(cell and cell.strip() for cell in rows_data[0]):
                        # Use first row as header if it contains values
                        columns = rows_data[0]
                        df = pd.DataFrame(rows_data[1:], columns=columns)
                    else:
                        # Create generic column names
                        columns = [f"Column {i+1}" for i in range(len(rows_data[0]))]
                        df = pd.DataFrame(rows_data, columns=columns)
                    
                    tables.append({
                        "page": page_num + 1,
                        "table_number": i + 1,
                        "dataframe": df
                    })
        
        return tables
    except Exception as e:
        st.error(f"Error extracting tables with PyMuPDF: {e}")
        return []

def get_cohere_summary(text, api_key):
    """Get summary of the text using Cohere API."""
    try:
        co = cohere.Client(api_key)
        response = co.summarize(
            text=text,
            length='medium',
            format='paragraph',
            model='command',
            additional_command='Make it comprehensive and highlight key points'
        )
        return response.summary
    except Exception as e:
        st.error(f"Error with Cohere API: {e}")
        return None

def main():
    st.title("📄 PDF Content Extractor")
    st.write("Upload a PDF file to extract images, tables, and text")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    cohere_api_key = st.sidebar.text_input("Cohere API Key (for summarization)", type="password")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        # Create tabs for different extraction types
        tab1, tab2, tab3, tab4 = st.tabs(["Text", "Images", "Tables", "Summary"])
        
        with tab1:
            st.header("Extracted Text")
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.text_area("Text Content", text, height=400)
                st.download_button("Download Text", text, file_name=f"{uploaded_file.name}_text.txt")
        
        with tab2:
            st.header("Extracted Images")
            uploaded_file.seek(0)  # Reset file pointer
            images = extract_images_from_pdf(uploaded_file)
            if images:
                cols = st.columns(3)
                for i, (img_name, img) in enumerate(images):
                    col = cols[i % 3]
                    with col:
                        st.image(img, caption=img_name, use_column_width=True)
                        
                        # Create download link for each image
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        btn = st.download_button(
                            label=f"Download {img_name}",
                            data=buf.getvalue(),
                            file_name=f"{uploaded_file.name}_{img_name}.png",
                            mime="image/png"
                        )
            else:
                st.info("No images found in the PDF.")
        
        with tab3:
            st.header("Extracted Tables")
            uploaded_file.seek(0)  # Reset file pointer
            tables = extract_tables_from_pdf_pymupdf(uploaded_file)
            if tables and len(tables) > 0:
                for table_info in tables:
                    page_num = table_info["page"]
                    table_num = table_info["table_number"]
                    df = table_info["dataframe"]
                    
                    st.subheader(f"Table {table_num} (Page {page_num})")
                    st.dataframe(df)
                    
                    # Download option for each table
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Table {table_num} from Page {page_num} as CSV",
                        data=csv,
                        file_name=f"{uploaded_file.name}_page{page_num}_table_{table_num}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No tables found in the PDF. Table detection may not work on all PDF types.")
                st.write("For complex tables, you might need to use alternative methods.")
        
        with tab4:
            st.header("Document Summary")
            if cohere_api_key:
                uploaded_file.seek(0)  # Reset file pointer
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    with st.spinner("Generating summary with Cohere..."):
                        summary = get_cohere_summary(text, cohere_api_key)
                        if summary:
                            st.write(summary)
                            st.download_button("Download Summary", summary, file_name=f"{uploaded_file.name}_summary.txt")
            else:
                st.info("Enter a Cohere API key in the sidebar to generate a summary.")

if __name__ == "__main__":
    main()