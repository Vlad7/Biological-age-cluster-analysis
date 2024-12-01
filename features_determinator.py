import info_enums as ie
import features as ft

def determine_features_all_and_features_selected(version, type):

    features_all = None
    features_selected = None

    if version == ie.NHANES.NHANES3_HDTrain:
        if type == ie.DatasetType.Antropometry:
            pass
        elif type == ie.DatasetType.Biochemistry:
            features_all = ft.features_NHANES3_HDTrain_biochemistry_all
            features_selected = ft.NHANES3_HDTrain_biochemistry_selected
        elif type == ie.DatasetType.Gemogramma:
            pass

    elif version == ie.GERONTOLOGY.NEW :
        if type == ie.DatasetType.Antropometry:
            pass
        elif type == ie.DatasetType.Biochemistry:
            features_all = ft.gerontology_biochemistry_all
            features_selected = features_all
        elif type == ie.DatasetType.Bones:
            features_all = ft.gerontology_bones_all
        elif type == ie.DatasetType.Gemogramma:
            features_all = ft.gerontology_gematology_all
            features_selected = features_all

    return features_all, features_selected