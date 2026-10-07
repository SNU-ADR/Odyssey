import json

# Please fill in your team information here
method = "method"  # <str> -- name of the method
team = "team"  # <str> -- name of the team, !!!identical to the Google Form!!!
authors = ["authors"]  # <list> -- list of str, authors
email = "email"  # <str> -- e-mail address
institution = "institution"  # <str> -- institution or company
country = "country"  # <str> -- country or region


def main():
    with open('output.json', 'r') as file:
        output_res = json.load(file)

    submission_content = {
        "method": method,
        "team": team,
        "authors": authors,
        "email": email,
        "institution": institution,
        "country": country,
        "results": output_res
    }

    with open('submission.json', 'w') as file:
        json.dump(submission_content, file, indent=4)

if __name__ == "__main__":
    main()
