import dataset_info as di
import features as ft

def determine_features_all_and_features_selected(version, type, hight_correlated_features):

    features_all = None
    features_selected = None

    dataset_features = []
    method = ""




    if version == di.NHANES.NHANES3_HDTrain:
        if type == di.DatasetType.Antropometry:
            pass
        elif type == di.DatasetType.Biochemistry:
            # Add all attributes from NHANES biochemistry
            features_all = ft.features_NHANES3_HDTrain_biochemistry_all
            features_selected = ft.NHANES3_HDTrain_biochemistry_selected

            dataset_features.extend(features_selected)
            print("All features: " + str(ft.NHANES3_HDTrain_biochemistry_selected))

            method += "polynomial"


        elif type == di.DatasetType.Gemogramma:
            pass

    elif version == di.GERONTOLOGY.NEW :
        if type == di.DatasetType.Antropometry:
            pass

        ### Працює
        elif type == di.DatasetType.Biochemistry:
            features_all = ft.gerontology_biochemistry_all
            features_selected = features_all

            # Add all attributes from biochemistry
            if hight_correlated_features == False:
                # Select feature labels that hight correlates with age
                dataset_features.extend((ft.gerontology_biochemistry_both_versions))
                dataset_features.extend((ft.gerontology_biochemistry_new_additionally))

                print("All features: " + str(ft.gerontology_biochemistry_both_versions +
                                             ft.gerontology_biochemistry_new_additionally))

            method += "linear"



        elif type == di.DatasetType.Bones:
            features_all = ft.gerontology_bones_all
        elif type == di.DatasetType.Gemogramma:
            features_all = ft.gerontology_gematology_all
            features_selected = features_all

    return features_all, features_selected

    if version == ie.GERONTOLOGY.NEW and datasettype == ie.DatasetType.Biochemistry:
        # Add all attributes from biochemistry
        if hight_correlated_features == False:
            # Select feature labels that hight correlates with age
            dataset_attributes.extend((ft.gerontology_biochemistry_both_versions))
            dataset_attributes.extend((ft.gerontology_biochemistry_new_additionally))

            print("All features: " + str(ft.gerontology_biochemistry_both_versions +
                                         ft.gerontology_biochemistry_new_additionally))

        method += "linear"

    elif version == ie.GERONTOLOGY.NEW and datasettype == ie.DatasetType.Bones:
        # Add all attributes from bones
        dataset_attributes.extend((ft.gerontology_bones_all))

        print("All features: " + str(ft.gerontology_bones_all))

        method += "polynomial"

    elif version == ie.GERONTOLOGY.NEW and datasettype == ie.DatasetType.Gemogramma:
        # Add all attributes from gematology
        if hight_correlated_features == False:
            dataset_attributes.extend((ft.gerontology_gematology_all))
            print("All features: " + str(ft.gerontology_gematology_all))

        else:
            # Select feature labels that hight correlates with age
            dataset_attributes.extend(ft.gerontology_gematology_hight_correlation_with_age)
            print("Selected features: " + str(ft.gerontology_gematology_hight_correlation_with_age))

        method += "polynomial"


