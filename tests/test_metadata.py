"""Проверки разбора метаданных JSON-LD/BattINFO набора Empa Aurora."""

import json

from pinn_soh.data.metadata import parse_metadata

MINIMAL_DOC = {
    "@context": "https://w3id.org/emmo/domain/battery/context",
    "@graph": [
        {
            "@type": "BatteryTest",
            "hasTestObject": {
                "@type": "CoinCell",
                "hasPositiveElectrode": {
                    "@type": "Electrode",
                    "hasCoating": {
                        "@type": "ElectrodeCoating",
                        "hasActiveMaterial": {
                            "rdfs:comment": "LithiumIronPhosphate",
                            "molecularFormula": {"rdfs:comment": "LiFePO4"},
                        },
                    },
                    "hasMeasuredProperty": [
                        {
                            "@type": "RatedCapacity",
                            "hasNumericalPart": {"hasNumberValue": 1.0},
                            "hasMeasurementUnit": "emmo:MilliAmpereHourPerSquareCentiMetre",
                        },
                        {
                            "@type": "Diameter",
                            "hasNumericalPart": {"hasNumberValue": 14},
                            "hasMeasurementUnit": "unit:MilliM",
                        },
                    ],
                },
                "hasNegativeElectrode": {
                    "@type": "Electrode",
                    "hasCoating": {
                        "@type": "ElectrodeCoating",
                        "hasActiveMaterial": {"@type": "Graphite"},
                    },
                },
                "hasElectrolyte": {
                    "@type": "OrganicElectrolyte",
                    "hasSolvent": {
                        "@type": "Solvent",
                        "hasConstituent": [
                            {
                                "@type": "EthyleneCarbonate",
                                "hasMeasuredProperty": {
                                    "@type": "VolumeFraction",
                                    "hasNumericalPart": {"hasNumberValue": 30},
                                    "hasMeasurementUnit": "unit:PERCENT",
                                },
                            }
                        ],
                    },
                },
            },
            "hasMeasurementParameter": {
                "@type": ["ConstantCurrentConstantVoltageCycling", "IterativeWorkflow"],
                "hasTask": {
                    "@type": "Charging",
                    "hasInput": [
                        {
                            "@type": "ElectricCurrentDensity",
                            "hasNumericalPart": {"hasNumberValue": 0.1},
                            "hasMeasurementUnit": "emmo:MilliAmperePerSquareCentiMetre",
                        },
                        {
                            "@type": ["UpperVoltageLimit", "TerminationQuantity"],
                            "hasNumericalPart": {"hasNumberValue": 4.2},
                            "hasMeasurementUnit": "emmo:Volt",
                        },
                    ],
                    "hasNext": {
                        "@type": "Discharging",
                        "hasInput": [
                            {
                                "@type": "ElectricCurrentDensity",
                                "hasNumericalPart": {"hasNumberValue": 0.1},
                                "hasMeasurementUnit": "emmo:MilliAmperePerSquareCentiMetre",
                            }
                        ],
                        "hasNext": {
                            "@type": "IterativeWorkflow",
                            "hasInput": [
                                {
                                    "@type": "NumberOfIterations",
                                    "hasNumericalPart": {"hasNumberValue": 5},
                                    "hasMeasurementUnit": "emmo:UnitOne",
                                }
                            ],
                            "hasTask": {
                                "@type": "Resting",
                                "hasInput": [
                                    {
                                        "@type": "Duration",
                                        "hasNumericalPart": {"hasNumberValue": 6},
                                        "hasMeasurementUnit": "emmo:Hour",
                                    }
                                ],
                            },
                        },
                    },
                },
            },
        }
    ],
}


def test_parse_minimal_metadata(tmp_path):
    path = tmp_path / "empa__ccid000001.metadata.json"
    path.write_text(json.dumps(MINIMAL_DOC))
    meta = parse_metadata(path)
    assert meta.cell_id == "empa__ccid000001"
    assert meta.cathode_material == "LiFePO4"
    assert meta.anode_material == "Graphite"
    assert meta.nominal_capacity_ah is not None
    modes = [step.mode for step in meta.protocol]
    assert modes[0] == "cc_charge" and "cc_discharge" in modes
    workflow = next(s for s in meta.protocol if s.mode == "workflow")
    assert workflow.repeat == 5
    assert workflow.steps[0].mode == "rest"
    assert workflow.steps[0].duration_s == 21600.0
    assert "EthyleneCarbonate" in meta.electrolyte


def test_parse_empty_document_does_not_fail(tmp_path):
    path = tmp_path / "empa__ccid000002.metadata.json"
    path.write_text("{}")
    meta = parse_metadata(path)
    assert meta.cathode_material is None
    assert meta.protocol == []


def test_unknown_nodes_go_to_raw_unmatched(tmp_path):
    doc = dict(MINIMAL_DOC)
    graph = [dict(MINIMAL_DOC["@graph"][0])]
    cell = dict(graph[0]["hasTestObject"])
    cell["mysteriousField"] = {"inner": 42}
    graph[0]["hasTestObject"] = cell
    doc["@graph"] = graph
    path = tmp_path / "empa__ccid000003.metadata.json"
    path.write_text(json.dumps(doc))
    meta = parse_metadata(path)
    assert "mysteriousField" in meta.raw_unmatched
