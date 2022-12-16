import streamlit as st
from pag_spine import main as pag_spine
from pag_burger import main as pag_burger
from pag_fritti import main as pag_fritti
from pag_cocktail import main as pag_cocktail
from pag_bar import main as pag_bar
from pag_multivariate import main as pag_multivariate


def main():
	
	################ load logo from web #########################
	from PIL import Image
	import requests
	from io import BytesIO
	url='https://static.wixstatic.com/media/bf9ae3_50bab6c255b64fe5a177c0d2215f0898~mv2.jpg/v1/fill/w_980,h_766,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/bf9ae3_50bab6c255b64fe5a177c0d2215f0898~mv2.jpg'
	response = requests.get(url)
	image = Image.open(BytesIO(response.content))
	st.title("FAV BIG DATALAB PW BiFor")
	st.image(image, caption='',use_column_width=True)

				
	pag_name = ["Spine","Burger","Fritti","Cocktail","Bar","Multivariate"]
	
	OPTIONS = pag_name
	st.subheader('Seleziona la categoria di interesse') 
	sim_selection = st.selectbox('',OPTIONS)

	if sim_selection == pag_name[0]:
		pag_spine()
	elif sim_selection == pag_name[1]:
		pag_burger()
	elif sim_selection == pag_name[2]:
		pag_fritti()
	elif sim_selection == pag_name[3]:
		pag_cocktail()
	elif sim_selection == pag_name[4]:
		pag_bar()
	elif sim_selection == pag_name[5]:
		pag_multivariate()
	else:
		st.markdown("Something went wrong. We are looking into it.")


if __name__ == '__main__':
	main()