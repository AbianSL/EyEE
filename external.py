# DELETE THIS AFTER USING IT

from astropy.coordinates import SkyCoord
import astropy.units as u

# Definir la coordenada sexagesimal
c = SkyCoord(ra='15h18m33.22s', dec='+02d04m51.7s', frame='icrs')

print(f"RA: {c.ra.deg} degrees")
print(f"Dec: {c.dec.deg} degrees")

from astroquery.vizier import Vizier

# Buscar el catálogo DAML02 en VizieR
result = Vizier.get_catalogs("B/daml02")
print(result)
