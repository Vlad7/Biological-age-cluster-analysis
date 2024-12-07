import dataset_info as di
import features as ft

def determine_features_all_and_features_selected(version, type):

    features_all = None
    features_selected = None

    if version == di.NHANES.NHANES3_HDTrain:
        if type == di.DatasetType.Antropometry:
            pass
        elif type == di.DatasetType.Biochemistry:
            features_all = ft.features_NHANES3_HDTrain_biochemistry_all
            features_selected = ft.NHANES3_HDTrain_biochemistry_selected
        elif type == di.DatasetType.Gemogramma:
            pass

    elif version == di.GERONTOLOGY.NEW :
        if type == di.DatasetType.Antropometry:
            pass

        ### Працює
        elif type == di.DatasetType.Biochemistry:
            features_all = ft.gerontology_biochemistry_all
            features_selected = features_all



        elif type == di.DatasetType.Bones:
            features_all = ft.gerontology_bones_all
        elif type == di.DatasetType.Gemogramma:
            features_all = ft.gerontology_gematology_all
            features_selected = features_all

    return features_all, features_selected