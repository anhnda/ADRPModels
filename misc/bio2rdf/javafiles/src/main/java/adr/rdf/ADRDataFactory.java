package adr.rdf;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdfconnection.*;
import org.apache.jena.system.Txn;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;

public class ADRDataFactory {
    HashSet<String> ValidDrugBankIDs;

    public ADRDataFactory(){
        this.ValidDrugBankIDs = new HashSet<String>();
    }
    public void loadValidDrugBankIDs(){
        String[] paths = {Config.AEOLUS_DATA_PATH, Config.LIU_DATA_PATH};
        for (String p : paths) {
            try {
                FileReader rd = new FileReader(p);
                BufferedReader br = new BufferedReader(rd);
                String line;
                while ((line = br.readLine()) != null) {
                    String drugBankID = line.split("\\|")[0];
                    this.ValidDrugBankIDs.add(drugBankID);
                }
            } catch (IOException e) {
                System.err.println(e);
            }
        }
    }

    public void exportBio2RDFDrugTriple(){


        try ( RDFConnection conn = RDFConnectionFactory.connect(Config.RDF_DATASET_URL) ) {
                    FileWriter fw = new FileWriter(Config.BIO2RDF_TRIPLE_DATA_PATH);
                    BufferedWriter bw = new BufferedWriter(fw);

                    Txn.executeWrite(conn, ()-> {
                        // Process results by row:
                        // Query for drugBankIds

                        //Drug as subject
                        for (String drugBankId : this.ValidDrugBankIDs) {
                            System.out.println("Querying drug: "+drugBankId);

                            String queryString = String.format(Config.QUERY_TEMPLATE_S, drugBankId);
                            conn.querySelect(queryString, (qs) -> {
                                // Resource p = qs.getResource("p") ;
                                RDFNode p = qs.get("p");
                                RDFNode o = qs.get("o");


                                try {
                                    bw.write(drugBankId+"\t"+p.toString()+"\t"+o.toString().replace("\r\n","")+"\n");
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            });

                            //Drug as object
                            queryString = String.format(Config.QUERY_TEMPLATE_O, drugBankId);
                            conn.querySelect(queryString, (qs) -> {
                                // Resource p = qs.getResource("p") ;
                                RDFNode p = qs.get("p");
                                RDFNode s = qs.get("s");

                                try {
                                    bw.write(drugBankId+"\t"+p.toString()+"\t"+s.toString().replace("\r\n","")+"\n");
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            });
                        }

                    }); ;
                    bw.close();

        }
        catch (Exception e){
            System.err.println(e);
        }
    }
    public void run(){
        this.loadValidDrugBankIDs();
        this.exportBio2RDFDrugTriple();
    }
    public static void main(String args[]){

        ADRDataFactory adrDataFactory = new ADRDataFactory();
        adrDataFactory.run();
    }


}

