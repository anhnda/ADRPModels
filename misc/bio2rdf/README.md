##This package contains Java source code to extract ChemBio descriptors from Bio2RDF database.

### Requirements:
Apache jena
- Download url: http://archive.apache.org/dist/jena/binaries/
- An Apache jena fuseki standalone server. (Version 2.0.0)
- An Apache jena client. (Version 2.13.0)


Bio2RDF data:
- drugbank.nq: http://download.bio2rdf.org/files/release/4/drugbank/drugbank.nq.gz
- sider-se.np: http://download.bio2rdf.org/files/release/4/sider/sider-se.nq.gz

These files are downloaded and extracted at $DATA_DIR (a custom folder)


### Run a jena fuseki server:
```
fuseki-server --loc=$CUSTOM_LOC /adr
```
RDF Fuseki server is available at http://localhost:3030/adr/

###Import Bio2RDF data



```
tdbloader --loc $CUSTOM_LOC $DATA_DIR/drugbank.nq
tdbloader --loc $CUSTOM_LOC $DATA_DIR/sider-se.nq

```




```

Reference

Muñoz, Emir, Vít Nováček, and Pierre-Yves Vandenbussche. "Using drug similarities for discovery of possible adverse reactions." In AMIA Annual Symposium Proceedings, vol. 2016, p. 924. American Medical Informatics Association, 2016.
