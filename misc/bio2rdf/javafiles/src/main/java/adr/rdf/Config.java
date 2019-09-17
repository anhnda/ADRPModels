package adr.rdf;

public class Config {
    public static String RDF_DATASET_PATH = "/media/anhnd/Storage/Data/rdf";
    public static String RDF_DATASET_URL = "http://localhost:3030/adr/";

    public static String DEFAULT_QUERY = "PREFIX DB: <http://bio2rdf.org/drugbank:> SELECT   ?p ?o FROM NAMED <http://bio2rdf.org/drugbank_resource:bio2rdf.dataset.drugbank.R4> FROM NAMED <http://bio2rdf.org/sider_resource:bio2rdf.dataset.sider.R4> WHERE { GRAPH ?g {DB:DB00220 ?p ?o}}";
    public static String QUERY_TEMPLATE_S = "PREFIX DB: <http://bio2rdf.org/drugbank:> " +
            "SELECT   ?p ?o " +
            "FROM NAMED <http://bio2rdf.org/drugbank_resource:bio2rdf.dataset.drugbank.R4> " +
            "FROM NAMED <http://bio2rdf.org/sider_resource:bio2rdf.dataset.sider.R4>" +
            " WHERE { GRAPH ?g {DB:%s ?p ?o}}";
    public static String QUERY_TEMPLATE_O = "PREFIX DB: <http://bio2rdf.org/drugbank:>" +
            " SELECT   ?s ?p " +
            " FROM NAMED <http://bio2rdf.org/drugbank_resource:bio2rdf.dataset.drugbank.R4>" +
            " FROM NAMED <http://bio2rdf.org/sider_resource:bio2rdf.dataset.sider.R4>" +
            " WHERE { GRAPH ?g {?s ?p DB:%s}}";

    public static String AEOLUS_DATA_PATH = "/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/AEOLUS_Data/AEOLUS_FinalDrugADR.tsv";
    public static String LIU_DATA_PATH = "/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/Liu_Data/ECFPLiuData.dat";
    public static String BIO2RDF_TRIPLE_DATA_PATH = "/home/anhnd/DTI Project/Codes/MethodsComparisonX/data/Bio2RDF/Bio2RDFDrugTriple.txt";
}
