SELECT
    td.chembl_id AS "Target_CHEMBL_ID",
    md.chembl_id AS "Compound_CHEMBL_ID",
    ac.pchembl_value,
    ass.chembl_id AS "Assay_CHEMBL_ID",
    ass.assay_type,
    ac.standard_type,
    ac.standard_value,
    ac.standard_units,
    ass.assay_tax_id AS "Assay Taxonomy",
    td.tax_id AS "TD Tax ID",
    ass.confidence_score,
    td.target_type,
    cr.src_compound_id,
    ass.src_assay_id,
    ass.src_id,
    src.src_description,
    ac.standard_relation,
    ac.activity_comment,
    ass.description,
    doc.year
FROM
    assays ass,
    activities ac,
    compound_records cr,
    molecule_dictionary md,
    target_dictionary td,
    organism_class oc,
    SOURCE src,
    docs doc
WHERE
    ass.assay_id = ac.assay_id AND ac.record_id = cr.record_id AND cr.molregno = md.molregno AND ass.tid = td.tid AND td.tax_id = oc.tax_id AND ass.src_id = src.src_id AND ac.doc_id = doc.doc_id AND td.tax_id IS NOT NULL AND ac.pchembl_value IS NOT NULL AND oc.l2 = "Mammalia" AND td.target_type IN("SINGLE PROTEIN") AND ass.assay_type IN("B") AND ac.standard_type IN(
        "Potency",
        "IC50",
        "EC50",
        "AC50",
        "Ki",
        "Kd",
        "XC50"
    ) AND ac.standard_units IN("nM", "uM", "M") AND ac.standard_relation IN("<", "=", "<<", "<=");