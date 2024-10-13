function updateOwnerFields(legalEntityField, targetField) {
    var legalEntityId = $(legalEntityField).val();
    console.log('Selected Legal Entity ID:', legalEntityId); // Debugging line
    if (legalEntityId) {
        var url = getUnitaOrganizzativeUrl.replace('LEGAL_ENTITY_ID', legalEntityId);
        console.log('AJAX URL:', url); // Debugging line
        $.ajax({
            url: url,
            success: function(data) {
                console.log('Data received:', data); // Debugging line
                var options = '<option value="">---------</option>';
                for (var i = 0; i < data.length; i++) {
                    options += '<option value="' + data[i].id + '">' + data[i].descrizione + '</option>';
                }
                $(targetField).html(options);
                $(targetField).val(''); // reset the value
            },
            error: function(xhr, status, errorThrown) {
                console.error('AJAX Error:', status, errorThrown); // Debugging line
            }
        });
    } else {
        console.log('No Legal Entity selected, clearing options'); // Debugging line
        $(targetField).html('<option value="">---------</option>');
        $(targetField).val(''); // reset the value
    }
}

$(document).ready(function() {
    console.log('Document is ready'); // Debugging line

    // Per il modello Processi
    $('#id_LegalEntity').change(function() {
        console.log('Legal Entity Changed'); // Debugging line
        updateOwnerFields('#id_LegalEntity', '#id_owner');
    });

    // Per il modello Documenti
    $('#id_legal_entity_controllante').change(function() {
        console.log('Legal Entity Controllante Changed'); // Debugging line
        updateOwnerFields('#id_legal_entity_controllante', '#id_owner_controllante');
    });

    $('#id_legal_entity_controllata').change(function() {
        console.log('Legal Entity Controllata Changed'); // Debugging line
        updateOwnerFields('#id_legal_entity_controllata', '#id_owner_controllata');
    });
});
