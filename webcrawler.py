import requests
from bs4 import BeautifulSoup
import numpy as np

class WebCrawler:

  def get_urls(self, base_url, limit):
    '''
    Este método utiliza <base_url> para encontrar las primeras <limit> urls de propiedades disponibles para una ubicación específica en el sitio "quierocasa.hn". Para realizar correctamente esta descarga se debe hacer repetidas solicitudes cambiando el valor de <nro_pagina>, como en el siguiente ejemplo:

    get_urls("https://www.quierocasa.hn/propiedad-en-venta-en-tegucigalpa/srp", 40)

    Al recibir <base_url> esta se modificará para hacer repetidas solicitudes a urls como:

    "https://www.quierocasa.hn/propiedad-en-venta-en-tegucigalpa/srp/page/<nro_pagina>"

    Este método retorna una lista de <limit> urls de propiedades encontradas (las primeras que aparezcan, sin repeticiones).Ej.:
    [
     "https://www.quierocasa.hn/se-vende-casa-en-el-hatillo-tegucigalpa/lldui3x/prd", 
     "https://www.quierocasa.hn/casa-en-venta-jardines-de-loarque-tegucigalpa/xhkvssb/prd", 
     ...
    ]
    '''
    urls = set() #Utilizamos un conjunto para evitar URL repetidas
    num_urls = 0
    num_pag = 1

    while num_urls < limit:
     url = base_url + "/page/" + str(num_pag)
     response = requests.get(url)
     doc = BeautifulSoup(response.text, "html.parser")
     for link in doc.find_all('a',{'title':'Ver detalles'}): #Obtenemos la casa (tag) donde coincide el title "Ver detalles"
      if num_urls == limit:
        break
      url = "https://www.quierocasa.hn" + link.get('href')
      attributes = self.get_attributes(url)
      if attributes['Tipo'] != "Casas" or attributes['Área'] == "" or attributes['Habitaciones'] == "" or attributes['Baños'] == "":
        continue
      urls.add(url)
      num_urls += 1
     num_pag += 1 
     #print('\n'.join(map(str, list(urls))))
    return list(urls)
    pass


  def get_attributes(self, property_url):
    '''
    Este método recibe una url de las que retorna el método get_urls, y la utiliza para revisar el código HTML que se obtiene al descargar dicha URL (puede usar la librería beatiful soup). A partir de dicho código se debe obtener un diccionario de atributos como el del ejemplo.

    Ejemplo de uso:
    get_attributes("https://www.quierocasa.hn/se-vende-casa-en-el-hatillo-tegucigalpa/lldui3x/prd")

    Retorna: 
    {
      'habitaciones': '3'
      'baños': '2.5'
      'área': '290'
      'precio': '275154'
      'tipo': 'casas'
      'amueblado': 'no'
      'cocina': '1'
      'salas de recepción': 'si'
      'balcón': '2'
      'tipo de calle': 'calle'
      'orientación': 'amanecer'
      'suelo': '1'
      'estacionamiento': '+5'
    }
    '''
    attributes = {"Habitaciones":'',"Baños":'',"Área":'',"Precio":'',"Tipo":'',"Amueblado":'',"Cocina":'',"Balcón":'',"Suelo":'',"Estacionamiento":''}
    response = requests.get(property_url)
    doc = BeautifulSoup(response.text, "html.parser")
    for x in attributes:
      if x == "Tipo":
        attribute = doc.find('div',{'class': 'font-weight-bold'})
        if not attribute == None:
          attribute = attribute.string
          attributes[x]=attribute
      elif x == "Precio":
        attribute = doc.find('span',{'class':'rs neg_price font-weight-semibold'})
        if not attribute == None:
          attribute = attribute.string.strip()
          attributes[x]=attribute
      elif x == "Habitaciones" or x == "Baños" or x == "Área":
        attribute = doc.find(string=str(x))
        if not attribute == None:
          parent = attribute.parent
          attribute = parent.find("span").string
          attribute = attribute.lower()
          attributes[x]=attribute.replace('+','').replace(' m2','').replace(' vrs²','').replace(' v2 de terreno','').replace(' Mts.','').replace(' mts2','').replace(' mts.','').replace(' de terreno','').replace(' de terreno.','')
      else:
        attribute = doc.find_all('p', {'class':'spec-title m-0'})
        if attribute == []:
          continue
        for detail in attribute:
          if x == detail.string:
            attribute = detail.parent.find("span").string
            attributes[x]=attribute.replace('+','')    
            break
          else:
            attribute = None
    # print(x + ":",attribute)
    #print("")
    #print(attributes)
    if attributes["Amueblado"] == "" or attributes["Amueblado"] == "No":
      attributes["Amueblado"] = 1
    else: 
      attributes["Amueblado"] = 2
    if attributes["Cocina"] == "":
      attributes["Cocina"] = 1
    if attributes["Balcón"] == "":
      attributes["Balcón"] = 1
    if attributes["Suelo"] == "":
      attributes["Suelo"] = 1
    if attributes["Estacionamiento"] == "":
      attributes["Estacionamiento"] = 1
    return attributes
    pass

  def proyect_dev(self):
    '''Este método retorna un diccionario que tiene como claves el número de cuenta y como valor el nombres completo de quien desarrollo el proyecto'''
    students = { 
     "20201000399": "Eduardo Josué Castro Arita"
    }
    return(students)
    pass

wc = WebCrawler()
# Obtener al menos 10 URLs de casas
urls = wc.get_urls("https://www.quierocasa.hn/propiedad-en-venta-en-tegucigalpa/srp", 350)
print(urls)
X = []
y = []
for url in urls:
    attributes = wc.get_attributes(url)
    #print(attributes)
    #print()
    if attributes["Precio"] is not None:
        y.append(float(attributes["Precio"].replace(",", "").replace("$", "")))  # Convertir el precio a float
        attributes.pop("Precio")# Excluir el atributo de precio
        attributes.pop("Tipo")
        X.append([float(val) for val in attributes.values()])
      
# Convertir los arreglos de python a arreglos de numpy
X = np.array(X)
y = np.array(y)
print()
print("Attributes of the houses: \n", X)
print()
print("House Prices: \n", y)
print()
print("Student: ", wc.proyect_dev())
# Guardar los arreglos de numpy en archivos
np.save("x.npy", X)
np.save("y.npy", y)